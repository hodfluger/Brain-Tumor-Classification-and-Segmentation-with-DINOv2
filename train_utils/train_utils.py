import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import torch.nn.functional as F
import random
import json
import pydensecrf.densecrf as dcrf
import scipy
from pydensecrf.utils import compute_unary, unary_from_softmax
import cv2

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
CLASS = 0
SEG = 1

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


########################################################################################################################
########################################################################################################################
class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)  # CE Loss

    def forward(self, logits, targets):
        """
        logits: (batch, 2, 224, 224)  -> Raw predictions before softmax/sigmoid
        targets: (batch, 224, 224)    -> Ground truth (0 or 1)
        """

        # Ensure correct type
        targets = targets.long()  # CrossEntropyLoss needs class indices
        logits = logits.float()   # Ensure logits are float

        # Compute CrossEntropy Loss
        ce_loss = self.ce_loss(logits, targets)

        # Compute Dice Loss
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # Take class-1 probabilities
        intersection = (probs * targets).sum(dim=(1, 2))
        union = (probs + targets).sum(dim=(1, 2))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()  # Mean over batch

        # Final loss
        loss = self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss
        return loss


########################################################################################################################
########################################################################################################################
#                                                   train_model
########################################################################################################################
########################################################################################################################

def train_model(type, model, criterion, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-3, models_dir='', title='', improv_cnt_th=10, save_score_th=0.9):
    if type == CLASS:
        metric = 'Accuracy'
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        metric = 'mIoU'
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    max_val_score = 0
    improv_cnt = 0
    train_loss_list = list()
    val_loss_list = list()
    train_score_list = list()
    val_score_list = list()

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sum_score = 0.0
        total_preds = 0

        # Training Loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if type == SEG:
                labels = labels.long()
                outputs = model(inputs, labels=labels)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                #loss = outputs.loss

            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if type == CLASS:
                _, predicted = torch.max(outputs, 1)
                sum_score += (predicted == labels).sum().item()
            else:
                labels = labels.squeeze()
                preds = outputs.argmax(dim=1)
                intersection = (preds * labels).sum(dim=(1, 2))
                union = (preds + labels).sum(dim=(1, 2)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                #print(iou.mean().item())
                sum_score += iou.sum().item()

            total_preds += labels.size(0)
        avg_train_score = sum_score / total_preds
        avg_train_loss = running_loss / len(train_loader)

        if type == CLASS:
            avg_val_loss, avg_val_score = evaluate_class_model(model, val_loader, device, criterion)
        else:
            avg_val_loss, avg_val_score = evaluate_seg_model(model, val_loader, device, criterion)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train {metric}: {avg_train_score:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val {metric}: {avg_val_score:.4f}")

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        train_score_list.append(avg_train_score)
        val_score_list.append(avg_val_score)

        if avg_val_score > max_val_score:
         best_model_weights = model.state_dict()
         score = np.round(100 * avg_val_score, 1)
         score_str = f'{score}'.replace('.', '_')
         max_val_score = np.copy(avg_val_score)
         train_data = [train_loss_list, val_loss_list, train_score_list, val_score_list]
         best_model_pt = f'{title}_{metric}{score_str}.pt'
         tmp_pt = models_dir.joinpath('tmp.pt')
         tmp_json = models_dir.joinpath('tmp.json')
         save_model(best_model_weights, tmp_pt, tmp_json, train_data)
         improv_cnt = 0
        else:
         improv_cnt += 1
         print(f'No improvement for {improv_cnt} epochs.')
         if improv_cnt == improv_cnt_th:
            break


    if max_val_score > save_score_th:
        best_model_name = models_dir.joinpath(best_model_pt)
        best_model_json = models_dir.joinpath(best_model_pt.replace('.pt', '.json'))
        save_model(best_model_weights, best_model_name, best_model_json, train_data)
    model.load_state_dict(best_model_weights)
    if type == CLASS:
        _, test_score = evaluate_class_model(model, test_loader, device, criterion)
    else:
        _, test_score = evaluate_seg_model(model, test_loader, device, criterion)
    print(f'Model {metric}: Val: {max_val_score}, Test {test_score}')
    return max_val_score, test_score

########################################################################################################################
########################################################################################################################
def evaluate_class_model(model, val_loader, device, criterion):
    # Validation Loop
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    avg_val_accuracy = val_correct_preds / val_total_preds
    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, avg_val_accuracy
########################################################################################################################
########################################################################################################################

def evaluate_seg_model(model, val_loader, device, criterion):
    # Validation Loop
    model.eval()
    val_loss = 0.0
    sum_iou_val = 0.0
    num_samples_val = 0.0

    model.eval()
    for imgs, labels in tqdm(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        labels = labels.squeeze(1)
        labels = labels.long()
        outputs = model(imgs, labels=labels)
        with torch.no_grad():
            loss = criterion(outputs.squeeze(), labels.squeeze())
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)

            intersection = (preds * labels).sum(dim=(1, 2))
            union = (preds + labels).sum(dim=(1, 2)) - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            sum_iou_val += iou.sum().item()
            num_samples_val += labels.size(0)

    avg_val_iou = sum_iou_val / num_samples_val
    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, avg_val_iou

########################################################################################################################
########################################################################################################################


def save_model(best_model_weigths, model_name, best_model_json, train_data):
    print('Saving Best Model...')
    torch.save(best_model_weigths,  model_name)
    with open(best_model_json, 'w') as f:
      json.dump(train_data, f)

########################################################################################################################
########################################################################################################################

def evaluate_trained_model(model, weights_path, val_loader, device, plot_confusion_matrix=False):
    weights = torch.load(weights_path, weights_only=True, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    val_correct_preds = 0
    val_total_preds = 0
    all_true_labels = []
    all_predicted_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            all_true_labels.extend(labels.cpu().numpy())
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)
            all_predicted_labels.extend(predicted.cpu().numpy())

    val_accuracy = val_correct_preds / val_total_preds
    print(f"Accuracy: {val_accuracy:.4f}")

    if plot_confusion_matrix:
        # Compute the confusion matrix
        label_map = {'Meningioma': 0, 'Glioma': 1, 'Pituitary': 2}
        display_labels = [key for key, value in sorted(label_map.items(), key=lambda item: item[1])]

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix_normalized,
            display_labels=display_labels,
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix [%]")
        plt.show()


########################################################################################################################
########################################################################################################################
def parse_model_name(model_name):
    seed = int(model_name.split('_')[0].split('seed')[1])
    dino_size = model_name.split('_')[2]
    batch_size = int(model_name.split('_')[3].split('bs')[1])
    crop = model_name.__contains__('Croped')
    return seed, dino_size, batch_size, crop

########################################################################################################################
########################################################################################################################
def plot_convergence_graph(type, json_path):
    if type == CLASS:
        metric = 'Accuracy'
    else:
        metric = 'mIoU'

    with open(json_path, 'r') as file:
        data = json.load(file)
        train_loss = data[0]
        val_loss = data[1]
        train_score = data[2]
        val_score = data[3]

    plt.subplot(1,2,1)
    epochs = np.arange(1, len(train_loss) + 1)
    plt.plot(epochs, np.asarray(train_loss), label='Train', color='r')
    plt.plot(epochs, np.asarray(val_loss), label='Validation', color='b')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    epochs = np.arange(1, len(train_score) + 1)
    plt.plot(epochs, np.asarray(train_score), label='Train', color='r')
    plt.plot(epochs, np.asarray(val_score), label='Validation', color='b')
    plt.title(metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()

    plt.show()

########################################################################################################################
########################################################################################################################
def bareaopen(mask):
    # Apply connectedComponentsWithStats to get labeled components and stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(mask))
    sizes = np.sort(stats[:, 4])

    # stats contains the following columns: [x, y, width, height, area]
    # We can filter components by their 'area', which is the last column in stats.

    # Set the minimum size (area) for a component to be kept
    if len(sizes) >= 2:
        min_size = sizes[-2] - 1
    else:
        min_size = 0

    # Create a new mask for the largest component only
    filtered_mask = np.zeros_like(mask)

    # Loop through each component (excluding the background which is label 0)
    for j in range(1, num_labels):
        area = stats[j, 4]
        if area >= min_size:
            filtered_mask[labels == j] = 1
    return filtered_mask

########################################################################################################################
########################################################################################################################

def create_segmented_image(im, pred, mask):
    rgb_im = np.stack((im,) * 3, axis=-1)
    r_channel = np.copy(im)
    r_channel[pred > 0] = 255
    g_channel = np.copy(im)
    g_channel[mask > 0] = 255
    rgb_im[:, :, 0] = r_channel
    rgb_im[:, :, 1] = g_channel

    rgb_orig = np.stack((im,) * 3, axis=-1)

    rgb_im = np.uint8(0.5 * rgb_orig + 0.5 * rgb_im)
    return rgb_im

########################################################################################################################
########################################################################################################################


def crf_mask(initial_mask, grayscale_image, spatial_kernel, apperance_kernel):

    rgb_image = np.stack([grayscale_image.astype(np.float32)] * 3, axis=-1)
    rgb_image_uint8 = np.clip(rgb_image, 0, 255).astype(np.uint8)



    inverse_mask = 1. - initial_mask
    mask =initial_mask[np.newaxis, :, :]
    inverse_mask = inverse_mask[np.newaxis, :, :]
    model_op = np.concatenate([inverse_mask, mask], axis=0)
    feat_first = model_op.reshape((2, -1))  # Flattening classes (e.g. BG and FG)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(grayscale_image.shape[1],grayscale_image.shape[0], 2)
    d.setUnaryEnergy(unary)  # add uniray potential to paiwise potential
    # Pairwise potential
    # it smooths the masks  # 5     10 sxy=(.5, .5)
    d.addPairwiseGaussian(sxy=spatial_kernel[0], compat=spatial_kernel[1],
                          kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)  # Spatial/Smootheness Kernel

    d.addPairwiseBilateral(sxy=apperance_kernel[0], srgb=apperance_kernel[1], rgbim=rgb_image_uint8,
                           # 5  13 10  sxy=(1.5, 1.5), srgb=(64, 64, 64)
                           compat=apperance_kernel[2], kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)  # Appearance/approximity Kernel

    Q = d.inference(3)
    CRF_op = np.argmax(Q, axis=0).reshape((grayscale_image.shape[0], grayscale_image.shape[1])).astype(np.float32)

    return CRF_op

########################################################################################################################
########################################################################################################################

def calc_f1(pred, target):

    TP = np.sum(np.logical_and(pred[:] > 0, target[:] > 0))
    FP = np.sum(np.logical_and(pred[:] > 0, target[:] == 0))
    FN = np.sum(np.logical_and(pred[:] == 0, target[:] > 0))

    # prec = min(TP / (TP + FP + 1e-6), 1)
    # recall = min(TP / (TP + FN + 1e-6), 1)

    F1 = 2 * TP / (2 * TP + FP + FN)
    return F1

########################################################################################################################
########################################################################################################################

def calc_iou(pred, mask):
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

########################################################################################################################
########################################################################################################################

def show_segmentations(model,batch_size, loader, device, spatial_kernel, apperance_kernel, plot=False, print_res=False, th=0.5):

    num_samples = 0.
    miou = 0.
    miou_post = 0.
    miou_crf = 0.
    f1_crf = 0.
    f1 = 0.

    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.to(device).squeeze(1).long()
        outputs = model(imgs, labels=labels)
        # preds = outputs.argmax(dim=1)

        for i in range(0, labels.size(0)):
            outputs_cpu = outputs[i].cpu().detach().numpy()
            outputs_cpu = scipy.special.softmax(outputs_cpu, axis=0)

            pred = np.asarray(outputs_cpu[1] > th)

            im = np.squeeze(imgs[i].cpu().numpy())[0]
            im = np.uint8(255 * (im - im.min()) / (im.max() - im.min()))
            mask = np.squeeze(labels[i].cpu().numpy())

            num_samples += 1

            filtered_pred = bareaopen(pred)

            crf_pred = crf_mask(filtered_pred, im, spatial_kernel=spatial_kernel, apperance_kernel=apperance_kernel)

            miou += calc_iou(pred, mask)
            f1 += calc_f1(pred, mask)
            f1_crf += calc_f1(crf_pred, mask)
            miou_post += calc_iou(filtered_pred, mask)
            miou_crf += calc_iou(crf_pred, mask)

            if plot:
                rgb_im = create_segmented_image(im, crf_pred, mask)
                plt.subplot(int(np.ceil(np.sqrt(batch_size))), int(np.ceil(np.sqrt(batch_size))), i + 1)
                plt.imshow(rgb_im, cmap='gray')
                plt.xticks([])
                plt.yticks([])
        if plot:
            plt.tight_layout()
            plt.show(block=True)

    if print_res:
        print(f'Only Model (Th={th}): mIoU={np.round(miou / num_samples, 3)},  F1={np.round(f1 / num_samples, 3)}')
        print(
            f'mIoU Model+Morph+CRF: mIoU={np.round(miou_crf / num_samples, 3)},  F1={np.round(f1_crf / num_samples, 3)}')





