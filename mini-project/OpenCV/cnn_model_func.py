"""
Animal image classification
- image data: ./data
- animal class: AFRICAN LEOPARD, CHEETA, LION, TIGER
---
- learning method: supervised learning, binary classification
- learning algorithm: CNN
- transfer learning model: vgg16
---
- frame work: Pytorch
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

import torchinfo
from torchinfo import summary

import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix

# utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_TRANSLATE = {0:'OTHERS', 1:'TIGER'}

def utils():
    """
    기본적인 사항을 확인하는 함수
    """
    print('----- Notice -----')
    print(f"device: {DEVICE}")
    print(f"label translate: {LABEL_TRANSLATE}\n")
    
    print(f"numpy ver: {np.__version__}")
    print(f"maatplotlib ver: {matplotlib.__version__}")
    # print(f"os ver: {os.__version__}")
    print(f"PIL ver: {PIL.__version__}\n")
    
    print(f"torch ver: {torch.__version__}")
    print(f"torchvision ver: {torchvision.__version__}")
    print(f"torchinfo ver: {torchinfo.__version__}")
    print(f"torchmetrics ver: {torchmetrics.__version__}")
    

# image transform function
