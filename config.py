# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import torch
from torchvision import transforms

# For load video
mask = 'data/mask_1920_1080.png'
video = 'data/parking_1920_1080.mp4'
step = 30
threshold_to_repeat_compute = 0.4

# For train model
DATASET = 'data/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKER = 4
IMAGE_SIZE = 35
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3
NUM_CLASS = 500
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "model_checkpoint.pt"
TRAIN_DIR = DATASET + 'train/'
TEST_DIR = DATASET + 'test/'
VAL_DIR = DATASET + 'val/'

"""Transform for train dataset"""
train_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                     transforms.RandomVerticalFlip(0.5),
                                     transforms.RandomHorizontalFlip(0.01),
                                     transforms.RandomGrayscale(0.05),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],)

"""Transform for val/test dataset"""
test_val_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),

                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],)