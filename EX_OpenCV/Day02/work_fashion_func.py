"""
Fashion MNIST class & function

 - learning method: supervised learning, multiclassification
 - learning algorithm: DNN, CNN
 
 - datasets: 'fashion_mnist_train.csv', 'fashion_mnist_test.csv', 이전에 과제했던 거 이번에는 내장 데이터셋 이용
 - features: pixel
 - label: 0 ~ 9
 - label translate: 0:T-Shirt, 1:Trouser, 2:Pullover, 3:Dress, 4:Coat,
                    5:Sandal, 6:Shirt, 7:Sneaker, 8:Bag, 9:Ankle Boot
 - frame work: Pytoch

"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import torchinfo, torchmetrics
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix
from torchinfo import summary

import torchvision

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 함수 선언 하기 전에 설정하는 거
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 38
torch.manual_seed(RANDOM_STATE)
LABEL_TRANSLATE = {0:'T-Shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat',
                    5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'}

# 기본 확인 하는 utils
def utils():
    """
    utils: package version, random_state, device
    ---
    - random_state
    - device
    - pandas version
    - numpy version
    - matplotlib version
    - scikit-learn version
    - torch version
    - torchinfo version
    - torchvision version
    - torchmetrics version
    """
    print('--- Notice ---')
    print(f"random_state: {RANDOM_STATE}")
    print(f"device: {DEVICE}\n")
    
    print(f"pandas ver: {pd.__version__}")
    print(f"numpy ver: {np.__version__}")
    print(f"matplotlib ver: {matplotlib.__version__}")
    print(f"scikit-learn ver: {sklearn.__version__}\n")
    
    print(f"torch ver: {torch.__version__}")
    print(f"torchinfo ver: {torchinfo.__version__}")
    print(f"torchvision ver: {torchvision.__version__}")
    print(f"torchmetrics ver: {torchmetrics.__version__}\n")
    
    
# fashion DNN model class
# ---------------------------------------------------------
# class perpose: 0 ~ 9 fashion mnist multi classification
# class name: FashionDNNModel
# parents class: nn.Module
# parameters: None
# attribute field: fc1, drop, fc2, fc3
# class function: create model structure, forward learning model
# class structure
# - fc1: input node: 784, output node: 256, activation function: ReLU
# - drop: 0.25
# - fc2: input node: 256, output node: 128, activation function: ReLU
# - fc3: input node: 128, output node: 10, activation function: None

class FashionDNNModel(nn.Module):
    """
    Fashion MNIST DNN multi classification model
    ---
    - parents class: nn.Module
    - parameters: None
    - attribute field: fc1, drop, fc2, fc3
    - class function: create model structure, forward learning model
    - class structure
        - fc1: input node: 784, output node: 256, activation function: ReLU
        - drop: 0.25
        - fc2: input node: 256, output node: 128, activation function: ReLU
        - fc3: input node: 128, output node: 10, activation function: None
    ---
    function
    - __init__()
    - forward()
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        y = x.view(-1, 784) # datasets에서 바로 텐서로 받을 거라서
        y = F.relu(self.fc1(y))
        y = self.drop(y)
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y
    

# fashion CNN model class
# ---------------------------------------------------------
# class perpose: 0 ~ 9 fashion mnist multi classification
# class name: FashionCNNModel
# parents class: nn.Module
# parameters: None
# attribute field: layer1, layer2 fc1, drop, fc2, fc3
# class function: create model structure, forward learning model
# class structure
# - layer1: conv2d(input:1, output:32, kernel:3*3, padding:1 to 0,), batch normalization, activate function: ReLU, maxpooling)
# - layer2: conv2d(input:32, output:64, kernel:3*3, padding:None), batch normalization, activate function: ReLU, maxpooling)
# - fc1: input node: 64*6*6, output node: 600, activate function: None
# - drop: 0.25
# - fc2: input node: 600, output node: 120, activate function: None
# - fc3: input node: 120, output node: 10, activate function: None

class FashionCNNModel(nn.Module):
    """
    Fashion MNIST DNN multi classification model
    ---
    - parents class: nn.Module
    - parameters: None
    - attribute field: layer1, layer2 fc1, drop, fc2, fc3
    - class function: create model structure, forward learning model
    - class structure
        - layer1: conv2d(input:1, output:32, kernel:3*3, padding:1 to 0,), batch normalization, activate function: ReLU, maxpooling)
        - layer2: conv2d(input:32, output:64, kernel:3*3, padding:None), batch normalization, activate function: ReLU, maxpooling)
        - fc1: input node: 64*6*6, output node: 600, activate function: None
        - drop: 0.25
        - fc2: input node: 600, output node: 120, activate function: None
        - fc3: input node: 120, output node: 10, activate function: None
    ---
    function
    - __init__()
    - forward()
    """
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64*6*6, 600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.fc3(y)
        
        return y
    

#  predict function
# -----------------------------------------------------
# - function name: predict
# - parameter: model, X_data
# - function return: loss, score_accuracy, score_f1score, proba
# -----------------------------------------------------
# must not update weight & bais

def predict(model, X_data):
    """
    Predict function

    Args:
        model (model instance): testing model
        X_data (tensor): feature tensor, just one tensor

    Returns:
        int: predicted fashion item
    """
    
    with torch.no_grad():
        pred = model(X_data).argmax().item()
        pred = LABEL_TRANSLATE[int(pred)]
        
    return pred


# validation & test function
# -----------------------------------------------------
# - function name: testing
# - parameter: model, x_data, y_data
# - function return: loss, score_accuracy, score_f1score
# -----------------------------------------------------
# must not update weight & bais

def testing(model, X_data, y_data):
    """
    Validation & Test function
    - pred is proba

    Args:
        model (model instance): testing model
        x_data (tensor): feature tensor
        y_data (tensor): label tensor

    Returns:
        tuple: loss, accuracy score, f1 score, proba
    """
    
    with torch.no_grad():
        pred = model(X_data)
        y_data = y_data.reshape(-1).long()
        loss = nn.CrossEntropyLoss()(pred, y_data)
        
        acc_score = Accuracy(task='multiclass', num_classes=10)(pred, y_data)
        f1_score = F1Score(task='multiclass', num_classes=10)(pred, y_data)
        
    return loss, acc_score, f1_score


# DNN model learning
# -------------------------------------------------------------
# - function name: DNNTraining
# - parameter: dataset, model, epochs, lr, batch_size, patience
# - function return: loss, accuracy score, f1 score, proba
# - optimizer: Adam
# - scheduler: ReduceLROnPlatea, standard: val_loss

def DNNTraining(dataset, model, epochs, lr=0.001, batch_size=32, patience=5):
    """
    DNN model training function
    - optimizer: Adam
    - scheduler: ReduceLROnPlatea, standard: train_loss

    Args:
        dataset (dataset instance): traing datasets
        model (model instance): model instance
        epochs (int): learning count
        lr (float, optional): learning rate. Defaults to 0.001.
        batch_size (int, optional): batch_size. Defaults to 32.
        patience (int, optional): model performance count threshold. Defaults to 5.

    Returns:
        tuple: loss list, accuracy list, f1-score list
    """
    loss_list = []
    acc_list = []
    f1_list = []
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    data_dl = DataLoader(dataset, batch_size=batch_size, drop_last=False)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    
    save_param = '../model/FashionMNIST/DNN_multi_clf_params.pth'
    save_model = '../model/FashionMNIST/DNN_multi_clf_model.pth'
    
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = 0, 0, 0
        
        for images, labels in data_dl:
            batch_cnt = len(dataset) / batch_size
            
            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)
            
            pred = model(train)
            
            loss = nn.CrossEntropyLoss()(pred, labels)
            total_t_loss += loss
            
            a_score = Accuracy(task='multiclass', num_classes=10)(pred, labels)
            total_t_acc += a_score
            f_score = F1Score(task='multiclass', num_classes=10)(pred, labels)
            total_t_f1 += f_score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = (total_t_loss/batch_cnt).item()
        train_acc = (total_t_acc/batch_cnt).item()
        train_f1 = (total_t_f1/batch_cnt).item()
        
        loss_list.append(train_loss)
        acc_list.append(train_acc)
        f1_list.append(train_f1)

        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]  loss: {train_loss:.6f}, score: {train_acc:.6f}")
            
        if len(acc_list) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_list[-1] >= max(acc_list):
                print("saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                
        scheduler.step(train_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]  loss: {train_loss:.6f}, score: {train_acc:.6f}")
            break
        
    return loss_list, acc_list, f1_list


# CNN model learning
# -------------------------------------------------------------
# - function name: CNNTraining
# - parameter: dataset, model, epochs, lr, batch_size, patience
# - function return: loss, accuracy score, f1 score, proba
# - optimizer: Adam
# - scheduler: ReduceLROnPlatea, standard: val_loss

def CNNTraining(dataset, model, epochs, lr=0.001, batch_size=32, patience=5):
    """
    CNN model training function
    - optimizer: Adam
    - scheduler: ReduceLROnPlatea, standard: train_loss

    Args:
        dataset (dataset instance): traing datasets
        model (model instance): model instance
        epochs (int): learning count
        lr (float, optional): learning rate. Defaults to 0.001.
        batch_size (int, optional): batch_size. Defaults to 32.
        patience (int, optional): model performance count threshold. Defaults to 5.

    Returns:
        tuple: loss list, accuracy list, f1-score list
    """
    loss_list = []
    acc_list = []
    f1_list = []
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    data_dl = DataLoader(dataset, batch_size=batch_size, drop_last=False)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    
    save_param = '../model/FashionMNIST/CNN_multi_clf_params.pth'
    save_model = '../model/FashionMNIST/CNN_multi_clf_model.pth'
    
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = 0, 0, 0
        
        for images, labels in data_dl:
            batch_cnt = len(dataset) / batch_size
            
            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)
            
            pred = model(train)
            
            loss = nn.CrossEntropyLoss()(pred, labels)
            total_t_loss += loss
            
            a_score = Accuracy(task='multiclass', num_classes=10)(pred, labels)
            total_t_acc += a_score
            f_score = F1Score(task='multiclass', num_classes=10)(pred, labels)
            total_t_f1 += f_score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = (total_t_loss/batch_cnt).item()
        train_acc = (total_t_acc/batch_cnt).item()
        train_f1 = (total_t_f1/batch_cnt).item()
        
        loss_list.append(train_loss)
        acc_list.append(train_acc)
        f1_list.append(train_f1)

        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]  loss: {train_loss:.6f}, score: {train_acc:.6f}")
            
        if len(acc_list) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_list[-1] >= max(acc_list):
                print("saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                
        scheduler.step(train_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]  loss: {train_loss:.6f}, score: {train_acc:.6f}")
            break
        
    return loss_list, acc_list, f1_list


# 그림 그리는 함수
def draw_two_plot(loss, score, title, type_):
    """
    draw loss & score

    Args:
        loss (_type_): _description_
        r2 (_type_): _description_
        title (_type_): _description_
    """
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss, label=f"train loss mean: {sum(loss)/len(loss):.6f}", color='#5587ED')
    ax2.plot(score, label=f"train score max: {max(score)*100:.2f} %", color='#00007F')
    
    fig.suptitle(f'{title} number {type_} multi classification', fontsize=15)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()
    
    
# predict figure draw
def draw_predict_figure(model, data_loader, num):
    """
    Draw predict figure in data set

    Args:
        model (model instane): predicting model
        data_loader (data_loader instance): data loder instance, must batch size is 1!!!
        num (int): data number
    """
    X_data, y_data = data_loader.dataset[num]
    pred = predict(model, X_data)

    print(f"predict fashion item: {LABEL_TRANSLATE[pred]}")
    print(f"real fashion item: {LABEL_TRANSLATE[y_data]}\n")

    image_data = X_data.reshape(-1, 28)
    print(f"image data: {image_data.shape}, {image_data.ndim}D")

    plt.imshow(image_data, cmap='BuPu')
    plt.title(f"[image - {LABEL_TRANSLATE[y_data]}]")
    plt.axis('off')
    plt.show()
    
    
def predict_show(pred, type_, num, data_loader):
    """
    show predict fashion item

    Args:
        pred (list): model predicted list
        label (list): real data list
        type_ (str): DNN or CNN
        num (int): showing image data number
        data_loader (data_loader instance): data loder instance, must batch size is 1!!!
    """
    image_data, label = data_loader.dataset[num]
    
    print(f"{type_} predict fashion item: {LABEL_TRANSLATE[pred[num]]}")
    print(f"real fashion item: {LABEL_TRANSLATE[label]}\n")

    image_data = image_data.reshape(-1, 28)
    print(f"image data: {image_data.shape}, {image_data.ndim}D")

    plt.imshow(image_data, cmap='Purples')
    plt.title(f"[image - {LABEL_TRANSLATE[label]}]")
    plt.axis('off')
    plt.show()