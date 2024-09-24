"""
MNIST class & function

 - learning method: supervised learning, multiclassification
 - learning algorithm: ANN, DNN
 
 - datasets: 'mnist_train.csv', 'mnist_test.csv'
 - features: pixel
 - label: 0 ~ 9
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import torchinfo, torchmetrics
from torchmetrics.classification import F1Score, Accuracy
from torchinfo import summary

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 함수 선언 하기 전에 설정하는 거
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 24
torch.manual_seed(RANDOM_STATE)

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
    print(f"torchmetrics ver: {torchmetrics.__version__}\n")
    

# model class
# ---------------------------------------------------------
# class perpose: 0 ~ 9 mnist multi classification
# class name: MnistMCModel
# parents class: nn.Module
# parameters: hidden_range, node_list
# attribute field: input_layer, hidden_layer, output_layer
# class function: create model structure, forward learning model
# class structure
# - input layer: input node: 784, output node: dynamic, activation function: ReLU
# - hidden layer: input node: dynamic, output node: dynamic, activation function: ReLU
# - output layer: input node: dynamic, output node: 10, activation function: None

class MnistMCModel(nn.Module):
    """
    MNIST multi classification model
    ---
    - parameters: hidden_range, node_list
    - attribute field: input_layer, hidden_layer, output_layer
    - class function: create model structure, forward learning model
    - class structure
        - input layer: input node: 784, output node: dynamic, activation function: ReLU
        - hidden layer: input node: dynamic, output node: dynamic, activation function: ReLU
        - output layer: input node: dynamic, output node: 10, activation function: None
    ---
    function
    - __init__()
    - forward() 
    
    """
    def __init__(self, hidden_range, node_list):
        super().__init__()
        
        self.input_layer = nn.Linear(784, node_list[0])
        self.hidden_layer = nn.ModuleList()
        for i in range(hidden_range):
            self.hidden_layer.append(nn.Linear(node_list[i], node_list[i+1]))
        self.output_layer = nn.Linear(node_list[-1], 10)
        
    def forward(self, x):
        
        y = F.relu(self.input_layer(x))
        for layer in self.hidden_layer:
            y = F.relu(layer(y))
        y = self.output_layer(y)
        
        return y
    

# train dataset class
# ------------------------------------------------------------
# dataset: 'mnist_train.csv'
# parents class: torch.utils.data.Dataset
# class name: TrainMnistDataset
# feature: pixel
# label: 0 ~ 9, label
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_train_ts, X_val_ts, y_train_ts, y_val_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class TrainMnistDataset(Dataset):
    """
    Train MNIST Dataset
    ---
    - parameter: data_df
    - attrubute: data_df, feature_df, label_df, X_train_ts, X_val_ts, y_train_ts, y_val_ts, n_rows, features, n_features, labels, n_labels
    ---
    function
    - __init__()
    - __len__()
    - __getitem__()
    """
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = data_df[data_df.columns[1:]]
        self.laber_df = data_df[[data_df.columns[0]]]
        
        # stratify: self.label_df
        # train : validation = 7 : 3
        X_train, X_val, y_train, y_val = train_test_split(self.feature_df, self.laber_df, stratify=self.laber_df, test_size=0.3, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        
        self.X_train_ts = torch.FloatTensor(X_train)
        self.X_val_ts = torch.FloatTensor(X_val)
        self.y_train_ts = torch.FloatTensor(y_train.values)
        self.y_val_ts = torch.FloatTensor(y_val.values)
        
        self.n_rows = X_train.shape[0]
        self.features = self.feature_df.columns
        self.n_features = len(self.features)
        self.labels = data_df[data_df.columns[0]].unique()
        self.n_labels = len(self.labels)
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        X_train_ts = self.X_train_ts[idx]
        y_train_ts = self.y_train_ts[idx]
        return X_train_ts, y_train_ts
    
# train dataset class
# ------------------------------------------------------------
# dataset: 'mnist_train.csv'
# parents class: torch.utils.data.Dataset
# class name: TrainMnistDataset
# feature: pixel
# label: 0 ~ 9, label
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_train_ts, X_val_ts, y_train_ts, y_val_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class TrainMnistDataset(Dataset):
    """
    Train MNIST Dataset
    ---
    - parameter: data_df
    - attrubute: data_df, feature_df, label_df, X_train_ts, X_val_ts, y_train_ts, y_val_ts, n_rows, features, n_features, labels, n_labels
    ---
    function
    - __init__()
    - __len__()
    - __getitem__()
    """
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = data_df[data_df.columns[1:]]
        self.laber_df = data_df[[data_df.columns[0]]]
        
        # stratify: self.label_df
        # train : validation = 7 : 3
        X_train, X_val, y_train, y_val = train_test_split(self.feature_df, self.laber_df, stratify=self.laber_df, test_size=0.3, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        
        self.X_train_ts = torch.FloatTensor(X_train)
        self.X_val_ts = torch.FloatTensor(X_val)
        self.y_train_ts = torch.FloatTensor(y_train.values)
        self.y_val_ts = torch.FloatTensor(y_val.values)
        
        self.n_rows = X_train.shape[0]
        self.features = self.feature_df.columns
        self.n_features = len(self.features)
        self.labels = data_df[data_df.columns[0]].unique()
        self.n_labels = len(self.labels)
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        X_train_ts = self.X_train_ts[idx]
        y_train_ts = self.y_train_ts[idx]
        return X_train_ts, y_train_ts


# test dataset class
# ------------------------------------------------------------
# dataset: 'mnist_test.csv'
# parents class: torch.utils.data.Dataset
# class name: TrainMnistDataset
# feature: pixel
# label: 0 ~ 9, label
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_test_ts, y_test_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class TestMnistDataset(Dataset):
    """
    Train MNIST Dataset
    ---
    - parameter: data_df
    - attrubute: data_df, feature_df, label_df, X_test_ts, y_test_ts, n_rows, features, n_features, labels, n_labels
    ---
    function
    - __init__()
    - __len__()
    - __getitem__()
    """
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = data_df[data_df.columns[1:]]
        self.laber_df = data_df[[data_df.columns[0]]]
        
        scaler = StandardScaler()
        scaler.fit(self.feature_df, self.laber_df)
        self.feature_df = scaler.transform(self.feature_df)
        
        self.X_test_ts = torch.FloatTensor(self.feature_df)
        self.y_test_ts = torch.FloatTensor(self.laber_df.values)
        
        self.n_rows = data_df.shape[0]
        self.features = data_df.columns[1:]
        self.n_features = len(self.features)
        self.labels = data_df[data_df.columns[0]].unique()
        self.n_labels = len(self.labels)
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        X_test_ts = self.X_test_ts[idx]
        y_test_ts = self.y_test_ts[idx]
        return X_test_ts, y_test_ts
    

# validation & test function
# -----------------------------------------------------
# - function name: testing
# - parameter: model, x_data, y_data
# - function return: loss, score_accuracy, score_f1score, proba
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


# predict function
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
        X_data (tensor): feature tensor

    Returns:
        int: predicted number
    """
    
    with torch.no_grad():
        pred = model(X_data).argmax().item()
        
    return int(pred)


# model learning
# -------------------------------------------------------------
# - function name: training
# - parameter: dataset, model, epochs, lr, batch_size, patience
# - function return: loss, accuracy score, f1 score, proba
# - optimizer: Adam
# - scheduler: ReduceLROnPlatea, standard: val_loss

def training(dataset, model, epochs, lr=0.001, batch_size=32, patience=10):
    """
    model training function
    - optimizer: Adam
    - scheduler: ReduceLROnPlatea, standard: val_loss

    Args:
        dataset (dataset instance): traing datasets
        model (model instance): model instance
        epochs (int): learning count
        lr (float, optional): learning rate. Defaults to 0.001.
        batch_size (int, optional): batch_size. Defaults to 32.
        patience (int, optional): model performance count threshold. Defaults to 10.

    Returns:
        tuple: loss dict, accuracy dict, f1-score dict
    """
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    f1_dict = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    data_dl = DataLoader(dataset, batch_size=batch_size, drop_last=False)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    
    save_param = '../model/MNIST/multi_clf_params.pth'
    save_model = '../model/MNIST/multi_clf_model.pth'
    
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = 0, 0, 0
        
        
        for X_train, y_train in data_dl:
            batch_cnt = dataset.n_rows / batch_size
            y_train = y_train.reshape(-1).long()
            
            pred = model(X_train)
            
            loss = nn.CrossEntropyLoss()(pred, y_train)
            total_t_loss += loss
            
            a_score = Accuracy(task='multiclass', num_classes=10)(pred, y_train)
            total_t_acc += a_score
            f_score = F1Score(task='multiclass', num_classes=10)(pred, y_train)
            total_t_f1 += f_score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        X_val, y_val = dataset.X_val_ts, dataset.y_val_ts
        val_loss, val_acc, val_f1 = testing(model, X_val, y_val)
        
        train_loss = (total_t_loss/batch_cnt).item()
        train_acc = (total_t_acc/batch_cnt).item()
        train_f1 = (total_t_f1/batch_cnt).item()
        
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
        f1_dict['train'].append(train_f1)
        f1_dict['val'].append(val_f1)

        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {val_acc.item():.6f}")
        
        if len(acc_dict['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_dict['val'][-1] > max(acc_dict['val']):
                print("saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                
        scheduler.step(val_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {val_acc.item():.6f}\n")
            break
        
        return loss_dict, acc_dict, f1_dict
    

# 그림 그리는 함수
def draw_two_plot(loss, score, title):
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
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(score['train'], label=f"train score max: {max(score['train'])*100:.2f} %", color='#00007F')
    ax2.plot(score['val'], label=f"validation score max: {max(score['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} number ANN multi classification', fontsize=15)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()