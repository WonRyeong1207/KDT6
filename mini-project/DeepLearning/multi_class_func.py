"""
animal image multi classification

- learning method: supervised learning, multi classification
- learning algorithm: CNN, undetermind

- datasets: 'https://www.kaggle.com/datasets/alessiocorrado99/animals10/data?select=raw-img' and maybe webcrawing?
- features: animal image
- labels: dog, horse, elephant, butterfly, chiken, cat, cow, sheep, spider, squirrel
- frame work: Pytorch

"""
# animal image multi classification

# learning method: supervised learning, multi calssification
# learning algorithm: CNN, undetermind

# datasets: https://www.kaggle.com/datasets/alessiocorrado99/animals10/data?select=raw-img' and maybe webcrawing?
# features: animal image
# labels: dog, horse, elephant, butterfly, chiken, cat, cow, sheep, spider, squirrel
# frame work: Pytorch

# module & package
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

import torchmetrics
from torchmetrics.classification import F1Score

import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torchinfo
from torchinfo import summary


# 함수 선언 하기 전에 설정하는 거
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 22   # 혹시나 사용을 한다면 의미는 있을 듯
torch.manual_seed(RANDOM_STATE)
# 기본 확인 하는 utils
def utils():
    """
    사용한 패키지나 모듈의 버전과 device, random_state 변수 값 확인
    """
    print('----- Notice -----')
    print(f"random_state: {RANDOM_STATE}")
    print(f"device: {DEVICE}\n")
    
    print(f"pandas ver: {pd.__version__}")
    print(f"numpy ver: {np.__version__}")
    print(f"matplotlib ver: {matplotlib.__version__}")
    print(f"sklearn ver: {sklearn.__version__}\n")
    
    print(f"torch ver: {torch.__version__}")
    print(f"torchmetrics ver: {torchmetrics.__version__}")
    print(f"tochvision ver: {torchvision.__version__}")
    print(f"torchinfo ver: {torchinfo.__version__}\n")
    

# make datasets function
# ------------------------------------------------------------
# dataset: './data/
# function name: make_dataset
# feature: animal image
# label: animal species [dog, horse, elephant, butterfly, chiken, cat, cow, sheep, spider, squirrel]
# function return: train_datasets, validation_datasets, test_datasets

def make_dataset():
    """
    make dataset function
    - dataset: './data'
    - features: animal image
    - labels: animal species [dog, horse, elephant, butterfly, chiken, cat, cow, sheep, spider, squirrel] to Italian language
    - function return: train_datasets, validation_datasets, test_datasets
    ---
    image transform
    - random horizantal flip: p=0.5
    - random vertical flip: p=0.5
    - random rotaion: degree=1
    - resize: (224, 224)
    - normalize: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] <- RGB coloer model base
    ---
    train, test, validation split ratio
    - trian : test = 8:2
    - train : validation = 8 : 2
    ---
    
    Returns:
        tuple: tensor datasets [train, test, validation]
    """

    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    img_datasets = ImageFolder(root='./data', transform=img_transforms)
    
    # train & test split
    # train : test : 8 : 2
    train_size = int(0.8 * len(img_datasets))
    test_size = len(img_datasets) - train_size
    train_datasets, test_datasets = random_split(img_datasets, [train_size, test_size])
    
    # train & validation split
    # train : validation = 8 : 2
    train_size = int(0.8 * len(train_datasets))
    val_size = len(train_datasets) - train_size
    train_datasets, val_datasets = random_split(train_datasets, [train_size, val_size])
    
    return train_datasets, test_datasets, val_datasets
    
    
# model class
# ----------------------------------------------------------------
# class perpose: language ainmal image & multi classification
# class name: VGG16Model
# parents class: nn.Module
# parameters: None or transfer learning?
# attribute field: input_layer, hidden_layer, output_layer
# class function: create structure, forward leanring model
# class structure
#   - input layer: input node 26, output node 20, activation function: ReLU
#   - hidden layer: input node 20, output node 15, activation function: ReLU
#   - hidden layer: input node 15, output node 10, activation function: ReLU
#   - output layer: input node 10, output node 4, activation function: None

# class LangMCModel(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.input_layer = nn.Linear(26, 20)
#         # self.hidden_layer_1 = nn.Linear(20, 15)
#         # self.hidden_layer_2 = nn.Linear(15, 10)
#         self.hidden_layer = nn.Linear(20, 10)
#         self.output_layer = nn.Linear(10, 4)
        
        
#     def forward(self, x):
#         y = F.relu(self.input_layer(x))
#         # y = F.relu(self.hidden_layer_1(y))
#         # y = F.relu(self.hidden_layer_2(y))
#         y = F.relu(self.hidden_layer(y))
#         y = self.output_layer(y)
        
#         return y
    
# create vgg16 model
class VGG16Model(nn.Module):
  def __init__(self):
    super(VGG16Model, self).__init__()  # transfer learning?
    self.model = nn.Sequential(

        # first block
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # second block
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # third block
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # fourth block
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        #fifth block
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # fully connected layers
        nn.Flatten(),

        nn.Linear(in_features=512*7*7, out_features=4096),
        nn.ReLU(),
        nn.Dropout(),

        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(),
        nn.Dropout(),

        nn.Linear(in_features=2048, out_features=80)

    )

  def forward(self, x):
    return self.model(x)
    
    
# vlidation & test function
# ----------------------------------------
# - function name: testing
# - parameter: model, x_data, y_data
# - function return: loss, score, pred
# ----------------------------------------
# must not update weight & bais
    
def testing(model, X_ts, y_ts):
    """
    validation & test function
    - must not update weight & bais

    Args:
        model (model instance): validation or testing model
        X_ts (tensor): validation or test feature tensor
        y_ts (tensor): validation or test label tensor

    Returns:
        tuple: tensor data [loss, score, prediction probability]
    """
    
    model.eval()
    with torch.no_grad():
        pred = model(X_ts)
        loss = nn.CrossEntropyLoss()(pred, y_ts)
        score = F1Score(task='multilabel', num_labels=10, num_classes=80)(pred, y_ts)
    
    return loss, score, pred

def predict(model, data_loader):
    """
    classification animal

    Args:
        model (model_instance):  testing model
        dataloader (tensor): testing feature & label tensor

    Returns:
        list: predict animal species
    """
    
    predicted_labels = []
    actual_labels = []

    model.eval()  # Set the model to evaluation mode
    with torch.inference_mode():  # Ensure no gradients are computed
        for images, labels in data_loader:
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
        
        
            labelDict = {0:'cane', 1:'cavallo', 2: 'elefante', 3:'fafalla', 4:'gollina',
                        5:'gatto', 6:'mucca', 7:'pecora', 8:'ragno', 9:'scoiattolo'}
            translate = {'cane':'dog', 'cavallo':'horse', 'elefante':'elephant', 'farflloa':'butterfly', 'gallina':'chiken',
                        'gatto':'cat', 'mucca':'cow', 'pecora':'sheep', 'ragno':'spider', 'scoiattolo':'squirrel'}
            result = []
            for pred in predicted_labels:
                result.append(translate[labelDict[pred]])
    
    return result

    
# model learning
# --------------------------------------
# - function name: training
# - parameter: dataloader?, model, epochs, lr, batch_size, threshold
# - function return: loss, score, pred
# - optimizer: optima.Adam

def training(model, train_loader, val_loader, epochs, lr, batch_size=32, threshold=0.0001):
    """
    model training function
    ---
    - optimizer algorithm: Adam

    Args:
        model (model instance): training multi classification
        train_loader (dataloader): train data
        val_loader (dataloader): valisdation data
        epochs (int): learning count
        lr (float): learning ratio
        batch_size (int): batch_size, Default to 32 
        threshold (int, optional): early stopping hyperparameter. Defaults to 0.0001

    Returns:
        tuple: train & validation loss & score values list in dict [loss, score, #prediction probability]
    """
    train_val_loss = {'train':[], 'val':[]}
    train_val_score = {'train':[], 'val':[]}
    train_val_pred = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    # data_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    save_param = f'./model/{model}_params.pth'
    save_model = f'./model/{model}.pth'
    # break_cnt = 0
    
    # optimization scheduler
    # patience: 5
    # otheers parameters: default
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min')   # validation loss
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='max')     # validation score
    
    model.train()
    for epoch in range(1, epochs+1):
        total_train_loss, total_train_score, total_train_pred = 0, 0, 0
        total_val_loss, total_val_score, total_val_pred = 0, 0, 0
        
        train_predicted_labels = []
        train_actual_labels = []
        
        # Training phase
        model.train()
        for images, labels in train_loader:
            # Move data to the appropriate device
            images = images
            labels = labels

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            train_pred = model(images)
            train_loss = nn.CrossEntropyLoss()(train_pred, labels)
            
            _, predicted = torch.max(train_pred, 1)
            train_predicted_labels.extend(predicted.cpu().numpy())
            train_actual_labels.extend(labels.cpu().numpy())
            train_score = f1_score(train_actual_labels, train_predicted_labels, average='weighted', zero_division=0)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item() * images.size(0)  # Accumulate loss
            # total_train_pred += train_pred.item()
            total_train_score += train_score

        # Average training loss for the epoch
        loss_train = total_train_loss / len(train_loader.dataset)
        # loss_train = loss_train / 16754
        # pred_train /= len(train_loader.dataset)
        score_train = total_train_score / len(train_loader.dataset)
        # score_train = score_train / 16754

        val_predicted_labels = []
        val_actual_labels = []
        # Validation phase
        model.eval()
        with torch.inference_mode():
            for val_images, val_labels in val_loader:
                # Move validation data to the appropriate device
                val_images = val_images
                val_labels = val_labels

                val_pred = model(val_images)
                val_loss = nn.CrossEntropyLoss()(val_pred, val_labels)
                
                _, predicted = torch.max(train_pred, 1)
                val_predicted_labels.extend(predicted.cpu().numpy())
                val_actual_labels.extend(labels.cpu().numpy())
                val_score = f1_score(train_actual_labels, train_predicted_labels, average='weighted', zero_division=0)
                
                total_val_loss += val_loss.item() * val_images.size(0)  # Accumulate loss
                # total_val_pred += val_pred.item() * val_images.size(0)
                total_val_score += val_score

        # Average validation loss for the epoch
        loss_val = total_val_loss / len(val_loader.dataset)
        # pred_val /= len(val_loader.dataset)
        score_val = total_val_score / len(val_loader.dataset)
        
            
        train_val_loss['train'].append(loss_train)
        train_val_loss['val'].append(loss_val)
        train_val_score['train'].append(score_train)
        train_val_score['val'].append(score_val)
        # train_val_pred['train'].append(pred_train)
        # train_val_pred['val'].append(pred_val)
            
        
        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {loss_train:.6f}, score: {score_val:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {loss_val:.6f}, score: {score_train:.6f}")
            
        
        # 학습 파라미터 저장
        if len(train_val_score['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        elif len(train_val_score['val']) > 1:
            if train_val_score['val'][-1] > max(train_val_score['val']):
                print("saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)

        # early stopping
        # if len(train_val_loss['val']) > 1:
        #     if train_val_loss['val'][-1] >= train_val_loss['val'][-2]:
        #         break_cnt += 1
        
        # stop
        # if break_cnt >= threshold:
        #     print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
        #     print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score:.6f}")
        #     print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {train_score:.6f}\n")
        #     break
        
        # optimization scheduler instance
        scheduler.step(loss_val)
        # scheduler.step(score_val)
        # print(f'scheduler.num_bad_epochs: {scheduler.num_bad_epochs}', end=' ')
        # print(f"scheduler.patience: {scheduler.patience}")
        
        # early stopping
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {loss_train:.6f}, score: {score_val:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {loss_val:.6f}, score: {score_train:.6f}")
            break
        
    return train_val_loss, train_val_score #, train_val_pred


# 그림 그리는 함수
def draw_two_plot(loss, r2, title):
    """ x축을 공유하고 y축을 따로 사용하는 함수

    Args:
        loss (dict): loss={'train':[], 'val':[]}
        r2 (dict): score={'train':[], 'val':[]}
        title (str): str (ex) loss & F1score
    """
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(r2['train'], label=f"train score max: {max(r2['train'])*100:.2f} %", color='#00007F')
    ax2.plot(r2['val'], label=f"validation score max: {max(r2['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} animal CNN multi classification', fontsize=15)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()
    






if __name__ == '__main__':
    utils()