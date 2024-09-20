# Language multi classification

# learning method: supervised learning, multi calssification
# learning algorithm: ANN

# train data: '../data/Language/lang_train.csv
# test data: '../data/Language/lang_test.csv
# feature: a to z
# label: en, fr, id, tl
# frame work: pytorch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import F1Score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 함수 선언 하기 전에 설정하는 거
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 19
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
# class perpose: language data learning & multi classification
# class name: LangMCModel
# parents class: nn.Module
# parameters: None
# attribute field: input_layer, hidden_layer, output_layer
# class function: create structure, forward leanring model
# class structure
#   - input layer: input node 26, output node 20, activation function: ReLU
#   - hidden layer: input node 20, output node 15, activation function: ReLU
#   - hidden layer: input node 15, output node 10, activation function: ReLU
#   - output layer: input node 10, output node 4, activation function: None

class LangMCModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_layer = nn.Linear(26, 20)
        # self.hidden_layer_1 = nn.Linear(20, 15)
        # self.hidden_layer_2 = nn.Linear(15, 10)
        self.hidden_layer = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, 4)
        
        
    def forward(self, x):
        y = F.relu(self.input_layer(x))
        # y = F.relu(self.hidden_layer_1(y))
        # y = F.relu(self.hidden_layer_2(y))
        y = F.relu(self.hidden_layer(y))
        y = self.output_layer(y)
        
        return y
    


# train dataset class
# ------------------------------------------------------------
# dataset: '../data/Language/lang_train.csv
# parents class: torch.utils.data.Dataset
# class name: LangTrainDataset
# feature: a ~ z
# label: language(en, fr, id, tl)
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_train_df, X_val_ts, y_train_df, y_val_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class LangTrainDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = self.data_df[self.data_df.columns[:-1]]
        label_sr = self.data_df['language']
        self.label_df = pd.get_dummies(label_sr).astype('int64')
        
        # stratify: self.label_df
        # train : val = 7 : 3
        X_train, X_val, y_train, y_val = train_test_split(self.feature_df, self.label_df, stratify=self.label_df, test_size=0.3, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        
        self.X_train_df = X_train
        self.X_val_ts = torch.FloatTensor(X_val)
        self.y_train_df = y_train
        self.y_val_ts = torch.FloatTensor(y_val.values)
        
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



# test dataset class
# ------------------------------------------------------------
# dataset: '../data/Language/lang_test.csv
# parents class: torch.utils.data.Dataset
# class name: LangTestDataset
# feature: a ~ z
# label: language(en, fr, id, tl)
# parameter: data_df
# attribute field: data_df, feature_df, label_df, X_test_ts, y_test_ts, n_rows, features, n_features, labels, n_lables
# class function: create dataset structure, length, get dataset bach size
# class structure
#   - __init__(self)
#   - __len__(self)
#   - __getitem__(self)

class LangTestDataset(Dataset):
    def __init__(self, data_df):
        super().__init__()
        
        self.data_df = data_df
        self.feature_df = self.data_df[self.data_df.columns[:-1]]
        label_sr = self.data_df['language']
        self.label_df = pd.get_dummies(label_sr).astype('int64')
        
        scaler = StandardScaler()
        scaler.fit(self.feature_df, self.label_df)
        self.feature_df = scaler.transform(self.feature_df)
        
        
        self.X_test_ts = torch.FloatTensor(self.feature_df)
        self.y_test_ts = torch.FloatTensor(self.label_df.values)
        
        self.n_rows = self.feature_df.shape[0]
        self.features = self.data_df.columns
        self.n_features = len(self.features)
        self.labels = self.data_df['language'].unique()
        self.n_labels = len(self.labels)
        
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        
        feature_ts = torch.FloatTensor(self.feature_df_df[idx])
        label_ts = torch.FloatTensor(self.label_df.iloc[idx].values)
        
        return feature_ts, label_ts
    
    
    
    # vlidation & test function
    # ----------------------------------------
    # - function name: testing
    # - parameter: model, x_data, y_data
    # - function return: loss, score, pred
    # ----------------------------------------
    # must not update weight & bais
    
def testing(model, X_ts, y_ts):
    with torch.no_grad():
        # y_ts = y_ts.reshape(-1).long()
        # y_ts = torch.argmax(y_ts, dim=1).long()
        
        pred = model(X_ts)
        
        # print(X_ts.shape, pred.shape, y_ts.shape)
        
        loss = nn.CrossEntropyLoss()(pred, y_ts)
        score = F1Score(task='multilabel', num_labels=4, num_classes=4)(pred, y_ts)
    
    return loss, score, pred

def predict(model, X_ts):
    with torch.no_grad():
        
        pred = model(X_ts).argmax()
        langDict = {0:'en', 1:'fr', 2: 'id', 3:'tl'}
        result = langDict[pred]
    
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
    
    save_param = '../model/language/lang_multi_clf.pth'
    save_model = '../model/language/lang_multi_clf_model.pth'
    break_cnt = 0
    
    for epoch in range(1, epochs+1):
        total_train_loss, total_train_score, total_train_pred = 0, 0, 0
        
        for X_train, y_train in data_dl:
            batch_cnt = dataset.n_rows / batch_size
            # y_train = y_train.reshape(-1).long()
            # y_train = torch.argmax(y_train, dim=1).long()
            
            pred = model(X_train)
            total_train_pred += pred
            
            # print(X_train.shape, pred.shape, y_train.shape)
            
            loss = nn.CrossEntropyLoss()(pred, y_train)
            total_train_loss += loss
            
            score = F1Score(task='multilabel', num_labels=4, num_classes=4)(pred, y_train)
            total_train_score += score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        X_val, y_val = dataset.X_val_ts, dataset.y_val_ts
        # print(X_val.shape, y_val.shape)
        val_loss, val_score, val_pred = testing(model, X_val, y_val)
        
        train_loss = (total_train_loss/batch_cnt).item()
        train_score = (total_train_score/batch_cnt).item()
        train_pred = (total_train_pred/batch_cnt)
        
        train_val_loss['train'].append(train_loss)
        train_val_loss['val'].append(val_loss)
        train_val_score['train'].append(train_score)
        train_val_score['val'].append(val_score)
        train_val_pred['train'].append(train_pred)
        train_val_pred['val'].append(val_pred)
        
        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {train_score:.6f}")
            
        
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
        if len(train_val_loss['val']) > 1:
            if train_val_loss['val'][-1] >= train_val_loss['val'][-2]:
                break_cnt += 1
        
        # stop
        if break_cnt >= threshold:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score:.6f}")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss.item():.6f}, score: {train_score:.6f}\n")
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
    
    fig.suptitle(f'{title} iris ANN multi classification', fontsize=15)
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()