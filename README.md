The entire project code (training models and evaluation) is implemented in train utils.py, models.py and datasets.py files, all located in the main project path). The pre-processed data is saved in the “MRI_data” folder.
The main platform for running is the notebook project.ipynb, which Contains 5 sections:

1. “Paths & Colab Configuration & Imports”:
The line with the comment
#CHANGE FOLDER NAME needs to be changed to the local drive path of the user.

2. “Classification - Train”:
This section enables training classification models. It runs a loop of training with different combinations of hyper-params. The hyperparams are defined in lists:

crop = False #True if we want to use “Tumor Cropped” images
batch_sizes = [16]
scales = [(1.0, 1.0)] # Random scaling as augmentation: (min scale, max scale). (1,0, 1,0) means no scaling.
learning_rates = ['5e-4']
loss_funcs = ['CE'] # for example: ['CE', 'Focaloss1', 'Focaloss2'] The number mentioned in Focaloss_ is gamma value
dino_size = 'base'
num_epochs = 100
All the possible combinations are taken, thus if we want a single training, every list should contain only one element. During training a json file contains the loss and accuracy per epoch is saved. The best model weights (highest validation accuracy) are saved as a pt file.

3. “Semantic Segmentation - Train”:
This section enables training semantic segmentation models. It runs a loop of training with different combinations of hyper-params. The hyperparams are defined in lists and dictionaries:

normalizations = [{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "name": 'ImageNet'}]  #also possible: {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "name": '05'}
batch_sizes = [20]
learning_rates = ['3e-4']
loss_funcs = ['Combined5'] #['CE', 'Combined2']  The number in "Combined_" is the alpha value, as described in the report.
num_epochs = 100
All the possible combinations are taken, thus if we want a single training, every list or dictionary should contain only one element. During training a json file contains the loss and mIoU per epoch is saved. The best model weights (highest validation accuracy) are saved as a pt file.

4. “Classification - Evaluation”:
This section evaluate a classification model, defined by its name (it assumes that the model pt file and json file are saved at the “saved models” folder and contains the following format:
model_name = 'seed100_CropedDino_base_bs16_lr5e-4_Focaloss1_scale10-10_val_Accuracy95_4.pt'
For example:
“Seed100” - The random seed used for training was 100.
“CropedDino” - the dataset used while training was the “Tumor Cropped” images.
“Base” - the DINOv2 model type.
“bs16” - natch size is 16.
“lr5e-4” - the learning rate was 0.0005
“Focaloss1” - The loss function used was focal loss with gamma=1.
“scale10-10” - as an augmentation the ransom scale params were (1.0, 1.0) - degenerated.
“Val_Accuracy95_4” - the achieved best validation accuracy.

This section evaluates the model on the test set, plots its confusion matrix and its convergence graphs (loss and accuracy), saved in a json file after training.

5. “Semantic Segmentation - Evaluation”:
This section evaluate a semantic segmentation model, defined by its name (it assumes that the model pt file and json file are saved at the “saved models” folder and contains the following format:
weights_path = 'seed100_DinoSeg_base_bs16_lr5e-4_Combined5_normImageNet_val_mIoU48_8.pt'
For example:
“Seed100” - The random seed used for training was 100.
“DinoSeg” - means that this is a segmentation model.
“Base” - the DINOv2 model type.
“bs16” = natch size is 16.
“lr5e-4” - the learning rate was 0.0005
“Combined1” - The loss function used was combined loss (CE+Dice) with alpha=0.5.
“NormImageNet” - The data was normalized due to ImageNet statistics.
“val_mIoU48.8” -  the achieved best validation mIoU was 48.8%.

This section evaluates the model on the test set, plots segmentation examples, and calculates the mIoU and F1 score before and after the CRF and morphological post processing. The user may also change the threshold for tumor detection (0.17 is the default):
detection_th = 0.17
