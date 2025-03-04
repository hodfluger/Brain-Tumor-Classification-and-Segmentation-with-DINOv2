import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
import cv2

CLASS = 0
SEG = 1

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)





#######################################################################################################################
#######################################################################################################################
def create_class_transforms(h_flip=0.5, translate=(0, 0.2), scale=(0.8, 1.4)):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(h_flip),
        transforms.RandomAffine(degrees=0, translate=translate, scale=scale),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return train_transform, test_transform

#######################################################################################################################
#######################################################################################################################

def create_seg_transforms(mean=None, std=None):
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor()])

    return img_transform, mask_transform

#######################################################################################################################
#######################################################################################################################
def calc_tumor_freq(data_loader):
    num_pixels = 224*224
    w_sum = 0.0
    cnt = 0.0
    for _, labels in data_loader:
        for i in range(labels.size(0)):
            label = labels[i].cpu().numpy()
            w = np.sum(label[:] > 0) / num_pixels
            w_sum += w
            cnt += 1
    print(w_sum / cnt)

#######################################################################################################################
#######################################################################################################################

class MRIDatasetClass(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # # Traverse the directory and collect image paths and labels
        # label_map = {'meningioma': 0, 'glioma': 1, 'pituitary': 2}

        image_paths = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]
        for f in image_paths:
            self.image_paths.append(os.path.join(image_dir, f))
            label_idx = int(f.split('label')[1].split('.png')[0])
            self.labels.append(label_idx - 1)



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, -1).astype(np.float32)
        image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
        image = Image.fromarray(image, mode="L")
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

#######################################################################################################################
#######################################################################################################################

def arrange_data(full_dataset, test_transform, batch_size, train_split=0.7, val_split=0.15):
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    if test_transform is not None:
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

#######################################################################################################################
#######################################################################################################################

def arrange_aplited_data(type, images_dir, masks_dir, transforms1, transforms2, batch_size, random_crop=0):
    if type == CLASS:
        train_dataset = MRIDatasetClass(image_dir=images_dir.joinpath('train'), transform=transforms1)
        val_dataset = MRIDatasetClass(image_dir=images_dir.joinpath('val'), transform=transforms2)
        test_dataset = MRIDatasetClass(image_dir=images_dir.joinpath('test'), transform=transforms2)
    else:
        train_dataset = MRIDatasetSeg(image_dir=images_dir.joinpath('train'), masks_dir=masks_dir.joinpath('train'), img_transform=transforms1, masks_transform=transforms2, random_crop=random_crop)
        val_dataset = MRIDatasetSeg(image_dir=images_dir.joinpath('val'), masks_dir=masks_dir.joinpath('val'), img_transform=transforms1, masks_transform=transforms2, random_crop=0)
        test_dataset = MRIDatasetSeg(image_dir=images_dir.joinpath('test'), masks_dir=masks_dir.joinpath('test'), img_transform=transforms1, masks_transform=transforms2, random_crop=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#######################################################################################################################
#######################################################################################################################

class MRIDatasetSeg(Dataset):
    def __init__(self, image_dir, masks_dir, img_transform=None, masks_transform=None, random_crop=0):
        self.image_dir = image_dir
        self.img_transform = img_transform
        self.masks_transform = masks_transform
        self.image_paths = []
        self.mask_paths = []
        self.random_crop = random_crop

        # # Traverse the directory and collect image paths and labels
        # label_map = {'meningioma': 0, 'glioma': 1, 'pituitary': 2}

        image_paths = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]
        for f in image_paths:
            self.image_paths.append(os.path.join(image_dir, f))
            self.mask_paths.append(os.path.join(masks_dir, f.replace('image', 'mask')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path, -1).astype(np.float32)
        image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        if self.random_crop > 0:
            image, mask = random_crop(image, mask, self.random_crop)

        # from matplotlib import pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(image, cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(mask, cmap='gray')
        # plt.show(block=True)

        mask = Image.fromarray(mask, mode="L")
        image = Image.fromarray(image, mode="L")


        # Apply transformations
        if self.img_transform:
            image = self.img_transform(image)
        if self.masks_transform:
            mask = self.masks_transform(mask)

        return image, mask


def random_crop(image, mask, crop_pr):
    heigth, width = mask.shape[0], mask.shape[1]
    i_pr = random.randint(0, crop_pr)
    j_pr = random.randint(0, crop_pr)
    i_i = int(float(i_pr) * heigth / 100)
    j_i = int(float(j_pr) * width / 100)

    i_pr = random.randint(0, crop_pr)
    j_pr = random.randint(0, crop_pr)
    i_f = heigth - int(float(i_pr) * heigth / 100)
    j_f = width - int(float(j_pr) * width / 100)

    image = image[i_i:i_f, j_i:j_f]
    mask = mask[i_i:i_f, j_i:j_f]
    return image, mask