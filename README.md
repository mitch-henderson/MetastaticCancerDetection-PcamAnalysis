# DenseNet Convolutional Neural Network for Metastatic Cancer Identification in Images

![RandomForest](https://github.com/mitch-henderson/densenet_convolutional_neural_network_applied_to_identify_metastatic_cancer_in_images_cspb3202-hw5/blob/main/2023_08_mitch___h_densenet_convolutional_neural_network_applied_to_iden.png)

In this competition, I create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. You may view and download the official Pcam dataset from GitHub https://github.com/basveeling/pcam. The data is provided under the CC0 License, following the license of Camelyon16.

## Histopathologic Cancer Detection
This project aims to detect metastatic cancer in small image patches taken from larger digital pathology scans. The dataset comes from the Pcam Kaggle competition.

## Data
The dataset contains 220,025 96x96 pixel RGB histopathology patches. 130,000 images are labeled as negative (no cancer) and 90,000 as positive (contains cancer). The data is split into training and test sets.

## Algorithm Toolset
The main tools and techniques used in this project:

- OpenCV for image loading and preprocessing
- Sklearn for train/validation split and model evaluation
- FastAI and PyTorch for the Deep Learning model and training
- DenseNet169 architecture as the base convolutional neural network
- Data augmentation (rotations, cropping etc.) to expand the training set
- Learning rate finder to find optimal learning rate
- Adam optimizer for training the model
- Binary cross-entropy loss function
- Validation set for hyperparameter tuning and model selection


## Model
A DenseNet169 CNN architecture pretrained on ImageNet is used as the base model. The model is trained for 1 epoch on the training set with a learning rate of 0.01302280556410551 and weight decay of 0.01. Data augmentation techniques like rotations, flipping, cropping etc. are used to expand the training set.

#### The final model achieves:

Accuracy: 92%
ROC AUC: 0.99
on the validation set.

## Usage
The train.py script trains the model on the training set and saves it to model.pth.

The predict.py script loads the trained model and makes predictions on the test set. It saves the predictions to submission.csv in the format accepted for the Kaggle competition.

### Installation
The code requires Python 3 and the following libraries:

matplotlib
opencv-python
pandas
scikit-learn
fastai
torchvision
The dependencies can be installed using:
``` pip install -r requirements.txt ```
#### References
The implementation is based on the following tutorial:

https://www.kaggle.com/code/awaisrauf/histopathologic-cancer-detection-with-fastai

##### License
The Pcam dataset is provided under the CC0 License, following the license of Camelyon16.
