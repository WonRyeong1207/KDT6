"""
Never News Clssification model & function

- learning method: supervised learning, multiclassification
- learning algorithm: RNN, LSTM

- datasets: '../data/news/
- features: peded rows?
- label: 0 ~ 7
- label translate: 0: 정치, 1: 경제, 2: 사회, 3: 생활/문화, 4:세계,
                   5: 기술/IT, 6: 연예, 7: 스포츠
- frame work: Pytorch
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchinfo, torchmetrics
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix
from torchinfo import summary

import string
import spacy
import soynlp
import konlpy
from konlpy.tag import Okt


# 함수 선언하기 전에 설정 하는 것
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 항상 설정하지만.. 안쓴다
RANDOM_STATE = 10
torch.manual_seed(RANDOM_STATE)
LABEL_TRANSLATE = {0:'정치', 1:'경제', 2:'사회', 3:'생활/문화', 4:'세계',
                   5:'기술/IT', 6:'연예', 7:'스포츠'}
DATA_ROOT = '../data/news/'
ko_model = 'ko_core_news_sm'
ko_spacy = spacy.load(ko_model)
ko_okt = Okt()
ko_soynlp = soynlp.word.WordExtractor()
punct = string.punctuation

# 기본적으로 확인하는 utils
def utils():
    """
    utils: package version, random_state, device
    ---
    - device
    - random_state
    ----
    - versions:
    pandas, numpy, matplotlib, torch, torchinfo, torchmetrics,
    spacy, konlpy, soynlp
    """
    print('----- Notice -----')
    print(f"device: {DEVICE}")
    print(f"random state: {RANDOM_STATE}\n")
    
    print(f"pandas ver: {pd.__version__}")
    print(f"numpy ver: {np.__version__}")
    print(f"matplotlib ver: {matplotlib.__version__}\n")
    
    print(f"torch ver: {torch.__version__}")
    print(f"torchinfo ver: {torchinfo.__version__}")
    print(f"torchmatrics ver: {torchmetrics.__version__}\n")
    
    print(f"spacy ver: {spacy.__version__}")
    print(f"soynlp ver: {soynlp.__version__}")
    print(f"konlpy ver: {konlpy.__version__}\n")
   

# load file list function
# ---------------------------------
# function name: load_file_list
# function parameters: data_root, num_range
# function return: file_list (dict)

def load_file_list(data_root='', num_range=8):
    """
    load file list to dict.

    Args:
        data_root (str): data root path
        num_range (int, optional): folder length. Defaults to 8.

    Returns:
        dict: file list
    """
    file_list_dict = {}
    for i in range(num_range):
        data_path = data_root+f'{i}/'
        file_list_dict[i] = os.listdir(data_path)
        
    return file_list_dict


# read file function
# ----------------------------------
# function name: read_file
# function parameters: data_root, file_dict
# function return: sentence (just one sentence)

def read_file(data_root, file_dict):
    
    pass