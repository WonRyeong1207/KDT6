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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import F1Score, Accuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 함수 선언 하기 전에 설정하는 거
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 12
torch.manual_seed(RANDOM_STATE)
# 기본 확인 하는 utils
def utils():
    print('--- Notice ---')
    print(f"random_state: {RANDOM_STATE}")
    print(f"device: {DEVICE}")
    print(f"random_state")
    print(f"torch ver: {torch.__version__}")
    print(f"pandas ver: {pd.__version__}")
    print(f"numpy ver: {np.__version__}")



# model class
# ----------------------------------------------------------------
# class perpose: language ainmal image & multi classification
# class name: DctMCModel
# parents class: nn.Module
# parameters: None or transfer learning?
# attribute field: input_layer, hidden_layer, output_layer
# class function: create structure, forward leanring model
# class structure
#   - input layer: input node 1024, output node 800, activation function: ReLU
#   - hidden layer: input node 800, output node 600, activation function: ReLU
#   - hidden layer: input node 600, output node 400, activation function: ReLU
#   - hidden layer: input node 400, output node 200, activation function: ReLU
#   - hidden layer: input node 200, output node 50, activation function: ReLU
#   - output layer: input node 50, output node 10, activation function: None

class DctMCModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_layer = nn.Linear(1024, 800)
        self.hidden_layer_1 = nn.Linear(800, 400)
        self.hidden_layer_2 = nn.Linear(400, 100)
        self.hidden_layer_3 = nn.Linear(100, 50)
        # self.hidden_layer_4 = nn.Linear(200, 50)
        self.output_layer = nn.Linear(50, 10)
        
        
    def forward(self, x):
        y = F.relu(self.input_layer(x))
        y = F.relu(self.hidden_layer_1(y))
        y = F.relu(self.hidden_layer_2(y))
        y = F.relu(self.hidden_layer_3(y))
        # y = F.relu(self.hidden_layer_4(y))
        y = self.output_layer(y)
        
        return y
    


# make datasets function
# ------------------------------------------------------------
# dataset: 'dct_image.csv'
# class name: DctDataset
# feature: animal image
# label: animal species [dog, horse, elephant, butterfly, chiken, cat, cow, sheep, spider, squirrel]
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_train_df, X_val_ts, X_test_ts, y_train_df, y_val_ts, y_test_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class DctDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = self.data_df[self.data_df.columns[:-1]]
        label_sr = self.data_df['animal']
        self.label_df = pd.get_dummies(label_sr).astype('int64')
        
        # stratify: self.label_df
        # train : test = 8 : 2
        # train : val = 8 : 2
        X_train, X_test, y_train, y_test = train_test_split(self.feature_df, self.label_df, stratify=self.label_df, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        self.X_train_df = X_train
        self.X_val_ts = torch.FloatTensor(X_val)
        self.X_test_ts = torch.FloatTensor(X_test)
        self.y_train_df = y_train
        self.y_val_ts = torch.FloatTensor(y_val.values)
        self.y_test_ts = torch.FloatTensor(y_test.values)
        
        self.n_rows = X_train.shape[0]
        self.features = self.feature_df.columns
        self.n_features = len(self.features)
        self.labels = self.data_df[self.data_df.columns[-1]].unique()
        self.n_labels = len(self.labels)
        
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        
        X_train_ts = torch.FloatTensor(self.X_train_df[idx])
        y_train_ts = torch.FloatTensor(self.y_train_df.iloc[idx].values)
        
        return X_train_ts, y_train_ts

    
    # vlidation & test function
    # ----------------------------------------
    # - function name: testing
    # - parameter: model, x_data, y_data
    # - function return: loss, score, pred
    # ----------------------------------------
    # must not update weight & bais
    
def testing(model, X_ts, y_ts):
    model.eval()
    with torch.no_grad():
        # y_ts = y_ts.reshape(-1).long()
        # y_ts = torch.argmax(y_ts, dim=1).long()
        
        pred = model(X_ts)
        
        # print(X_ts.shape, pred.shape, y_ts.shape)
        
        loss = nn.CrossEntropyLoss()(pred, y_ts)
        score = Accuracy(task='multilabel', num_labels=10, num_classes=10)(pred, y_ts)
    
    return loss, score, pred

def predict(model, X_ts):
    
    # X_ts = torch.FloatTensor(X_ts)
    model.eval()
    with torch.no_grad():
        
        pred = model(X_ts).argmax().item()
        animalDict = {0:'dog', 1:'horse', 2:'elephant', 3:'butterfly', 4:'chiken', 5:'cat', 6:'cow', 7:'sheep', 8:'spider', 9:'squirrel'}
        result = animalDict[pred]
    
    return result

    
# model learning
# --------------------------------------
# - function name: training
# - parameter: dataset, model, epochs, lr, batch_size, threshold
# - function return: loss, score, pred
# - optimizer: optima.Adam

def training(dataset, model, epochs, lr, batch_size, threshold):
    train_val_loss = {'train':[], 'val':[]}
    train_val_score = {'train':[], 'val':[]}
    train_val_pred = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    data_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    save_param = './model/dct_multi_clf_params.pth'
    save_model = './model/dct_multi_clf.pth'
    
    # optimization scheduler
    # patience: threshold
    # otheers parameters: default
    # scheduler = ReduceLROnPlateau(optimizer, patience=threshold, mode='min')   # validation loss
    scheduler = ReduceLROnPlateau(optimizer, patience=threshold, mode='max')     # validation score
    
    for epoch in range(1, epochs+1):
        total_train_loss, total_train_score, total_train_pred = 0, 0, 0
        
        model.train()
        for X_train, y_train in data_dl:
            batch_cnt = dataset.n_rows / batch_size
            # y_train = y_train.reshape(-1).long()
            # y_train = torch.argmax(y_train, dim=1).long()
            
            pred = model(X_train)
            total_train_pred += pred
            
            # print(X_train.shape, pred.shape, y_train.shape)
            
            loss = nn.CrossEntropyLoss()(pred, y_train)
            total_train_loss += loss
            
            score = Accuracy(task='multilabel', num_labels=10, num_classes=10)(pred, y_train)
            total_train_score += score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        X_val, y_val = dataset.X_val_ts, dataset.y_val_ts
        # print(X_val.shape, y_val.shape)
        val_loss, val_score, val_pred = testing(model, X_val, y_val)
        
        train_loss = (total_train_loss/batch_cnt).item()
        train_score = (total_train_score/batch_cnt).item()
        train_pred = (total_train_pred/batch_cnt)
        
        train_val_loss['train'].append(train_loss)
        train_val_loss['val'].append(val_loss.item())
        train_val_score['train'].append(train_score)
        train_val_score['val'].append(val_score.item())
        train_val_pred['train'].append(train_pred)
        train_val_pred['val'].append(val_pred)
        
        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {val_score:.6f}")
            
        
        # 학습 파라미터 저장
        if len(train_val_score['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if train_val_score['val'][-1] > max(train_val_score['val']):
                print("saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)

        # optimization scheduler instance
        # scheduler.step(val_loss)
        scheduler.step(val_score)
        # print(f'scheduler.num_bad_epochs: {scheduler.num_bad_epochs}', end=' ')
        # print(f"scheduler.patience: {scheduler.patience}")
        
        # early stopping
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {val_score:.6f}\n")
            break
        
    return train_val_loss, train_val_score, train_val_pred




# 그림 그리는 함수
def draw_two_plot(loss, r2, title):
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(r2['train'], label=f"train score max: {max(r2['train'])*100:.2f} %", color='#00007F')
    ax2.plot(r2['val'], label=f"validation score max: {max(r2['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} aniaml ANN multi classification', fontsize=15)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()