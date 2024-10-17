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
import os
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

import torchmetrics
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix

import sklearn
from sklearn.metrics import f1_score

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torchinfo
from torchinfo import summary


# 함수 선언 하기 전에 설정하는 거
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    img_datasets = ImageFolder(root='../data/raw-img', transform=img_transforms)
    
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

def make_dataset_cpu():
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
    img_datasets = ImageFolder(root='./data/', transform=img_transforms)
    
    # train & test split
    # train : test = 9 : 1
    train_size = int(0.9 * len(img_datasets))
    test_size = len(img_datasets) - train_size
    train_datasets, test_datasets = random_split(img_datasets, [train_size, test_size])
    
    # train & validation split
    # train : validation = 9 : 1
    train_size_2 = int(0.9 * len(train_datasets))
    val_size = len(train_datasets) - train_size_2
    train_datasets, val_datasets = random_split(train_datasets, [train_size_2, val_size])
    
    return train_datasets, test_datasets, val_datasets

class DctDataset(Dataset):
    def __init__(self):
        super().__init__()
        
        # self.data_df = data_df
        # self.feature_df = self.data_df[self.data_df.columns[:-1]]
        # label_sr = self.data_df['animal']
        # self.label_df = pd.get_dummies(label_sr).astype('int64')
        
        img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        img_datasets = ImageFolder(root='./data/', transform=img_transforms)
        
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
        
        # train_datasets = torch.FloatTensor(train_datasets)
        # val_datasets = torch.FloatTensor(val_datasets)
        # test_datasets = torch.FloatTensor(test_datasets)
    
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.val_datasets = val_datasets
        
        
    def __len__(self):
        # 데이터셋 길이를 반환
        return len(self.train_datasets)

    def __getitem__(self, idx):
        # train_datasets에서 인덱스를 기반으로 이미지와 라벨을 가져옴
        image, label = self.train_datasets[idx]
        
        return image, label


    
# model class
# ----------------------------------------------------------------
# class perpose: language ainmal image & multi classification
# class name: VGG16Model
# parents class: nn.Module
# parameters: None or transfer learning?
# attribute field: input_layer, hidden_layer, output_layer
# class function: create structure, forward leanring model
    
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


# transfer vgg16 model
class CustomVgg16MCModel(nn.Module):
    def __init__(self):
        super(CustomVgg16MCModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = self.vgg16.features
        self.avgpool = self.vgg16.avgpool
        self.classifier = self.vgg16.classifier
        self.custom_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def forward(self, x):
        y = self.features(x)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        y = self.custom_layer(y)
        return y

    
# vlidation & test function
# ----------------------------------------
# - function name: testing
# - parameter: model, x_data, y_data
# - function return: loss, score, pred
# ----------------------------------------
# must not update weight & bais
    
def testing_cuda(model, X_ts, y_ts):
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
        score = Accuracy(task='multilabel', num_labels=10, num_classes=80)(pred, y_ts)
    
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
            images = images.to(DEVICE,dtype=torch.float32)
            labels = labels.to(DEVICE,dtype=torch.float32)
            print(type(images), type(labels))
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
        
        
            labelDict = {0:'cane', 1:'cavallo', 2: 'elefante', 3:'farfalla', 4:'gallina',
                        5:'gatto', 6:'mucca', 7:'pecora', 8:'ragno', 9:'scoiattolo'}
            translate = {'cane':'dog', 'cavallo':'horse', 'elefante':'elephant', 'farfalla':'butterfly', 'gallina':'chiken',
                        'gatto':'cat', 'mucca':'cow', 'pecora':'sheep', 'ragno':'spider', 'scoiattolo':'squirrel'}
            result = []
            for pred in predicted_labels:
                result.append(translate[labelDict[pred]])
    
    return result, actual_labels

def predict_web(model, image_tensor):
    """
    classification animal for a single image

    Args:
        model (model_instance):  testing model
        image_tensor (tensor): transformed single image tensor

    Returns:
        str: predicted animal species
    """
    
    model.eval()  # Set the model to evaluation mode
    with torch.inference_mode():  # Ensure no gradients are computed
        image_tensor = image_tensor.to(DEVICE, dtype=torch.float32)  # 단일 이미지 텐서로 처리
        outputs = model(image_tensor)  # 모델에 이미지 입력
        _, predicted = torch.max(outputs, 1)  # 최대 값의 인덱스를 예측값으로
        
        labelDict = {0: 'cane', 1: 'cavallo', 2: 'elefante', 3: 'farfalla', 4: 'gallina',
                     5: 'gatto', 6: 'mucca', 7: 'pecora', 8: 'ragno', 9: 'scoiattolo'}
        translate = {'cane': 'dog', 'cavallo': 'horse', 'elefante': 'elephant', 'farfalla': 'butterfly', 'gallina': 'chicken',
                     'gatto': 'cat', 'mucca': 'cow', 'pecora': 'sheep', 'ragno': 'spider', 'scoiattolo': 'squirrel'}
        
        predicted_label = predicted.cpu().numpy()[0]  # 예측된 라벨 인덱스 추출
        predicted_animal = translate[labelDict[predicted_label]]  # 인덱스를 통해 동물 이름 변환
    
    return predicted_animal


    
# model learning
# --------------------------------------
# - function name: training
# - parameter: dataloader?, model, epochs, lr, batch_size, threshold
# - function return: loss, score, pred
# - optimizer: optima.Adam

def training_cuda(model, train_loader, val_loader, epochs, lr, batch_size=32, threshold=0.0001):
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
    
    save_param = f'gvv16_model_epoch100_params.pth'
    save_model = f'gvv16_model_epoch100.pth'
    # break_cnt = 0
    
    # optimization scheduler
    # patience: 5
    # otheers parameters: default
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min')   # validation loss
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='max')     # validation score

    # AMP를 사용하여 Mixed Precision Training을 구현
    scaler = GradScaler()
    
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
            images = images.to(DEVICE, dtype=torch.float16)
            labels = labels.to(DEVICE)

            # Forward pass
            # train_pred = model(images)
            # train_loss = nn.CrossEntropyLoss()(train_pred, labels)
            
            # _, predicted = torch.max(train_pred, 1)
            # train_predicted_labels.extend(predicted.cpu().numpy())
            # train_actual_labels.extend(labels.cpu().numpy())
            # train_score = f1_score(train_actual_labels, train_predicted_labels, average='weighted', zero_division=0)

            optimizer.zero_grad()

             # autocast를 통해 float16와 float32 혼합 연산을 관리
            with autocast():
                train_pred = model(images)
                train_loss = nn.CrossEntropyLoss()(train_pred, labels)

                _, predicted = torch.max(train_pred, 1)
                train_predicted_labels.extend(predicted.cpu().numpy())
                train_actual_labels.extend(labels.cpu().numpy())
                train_score = f1_score(train_actual_labels, train_predicted_labels, average='weighted', zero_division=0)

            # Backward pass에서 GradScaler 사용
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # # Clear gradients
            # optimizer.zero_grad()
            # # Backward pass
            # train_loss.backward()
            # optimizer.step()

            total_train_loss += train_loss.item() * images.size(0)  # Accumulate loss
            # total_train_pred += train_pred.item()
            total_train_score += train_score

            # Forward 및 Backward pass 후 메모리 해제
            torch.cuda.empty_cache()

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
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    # Move validation data to the appropriate device
                    val_images = val_images.to(DEVICE, dtype=torch.float16)
                    val_labels = val_labels.to(DEVICE)
                    
                    # val_pred = model(val_images)
                    # val_loss = nn.CrossEntropyLoss()(val_pred, val_labels)
                    
                    # _, predicted = torch.max(train_pred, 1)
                    # val_predicted_labels.extend(predicted.cpu().numpy())
                    # val_actual_labels.extend(labels.cpu().numpy())
                    # val_score = f1_score(train_actual_labels, train_predicted_labels, average='weighted', zero_division=0)

                    with autocast():
                        val_pred = model(images)
                        val_loss = nn.CrossEntropyLoss()(train_pred, labels)

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
            if train_val_score['val'][-1] >= max(train_val_score['val']):
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


def testing(model, test_images, test_labels):
    # test_labels = test_labels.unsqueeze(1).float()  # 차원이 맞지 않기 때문에 차원 추가
    # test_labels = torch.nn.functional.one_hot(test_labels, num_classes=10).float()

    
    with torch.no_grad():
        pred = model(test_images)
        loss = nn.CrossEntropyLoss()(pred, test_labels)
        acc = MulticlassAccuracy(num_classes=10)(pred, test_labels)
        f1 = MulticlassF1Score(num_classes=10)(pred, test_labels)
        mat = MulticlassConfusionMatrix(num_classes=10)(pred, test_labels)
        
    return loss, acc, f1, mat


def training(model, train_datasets, val_datasets, epochs=100, lr=0.001, batch_size=64, patience=5):
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    f1_dict = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    train_data_dl = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_data_dl = DataLoader(val_datasets, batch_size=100, shuffle=True)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    
    save_param = './model/CNN_mc_custom_clf_params.pth'
    save_model = './model/CNN_mc_custom_clf_model.pth'
    
    train_batch_cnt = len(train_datasets) / batch_size
    val_batch_cnt = len(val_datasets) / 100
    
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = 0, 0, 0
        total_v_loss, total_v_acc, total_v_f1 = 0, 0, 0
        
        model.train()
        for images, labels in train_data_dl:
            # batch_cnt = len(train_datasets) / batch_size
            # labels = labels.unsqueeze(1).float()
            # labels = torch.nn.functional.one_hot(labels, num_classes=10).float()

            pred = model(images)
            
            loss = nn.CrossEntropyLoss()(pred, labels)
            total_t_loss += loss
            
            acc = MulticlassAccuracy(num_classes=10)(pred, labels)
            total_t_acc += acc
            
            score = MulticlassConfusionMatrix(num_classes=10)(pred, labels)
            total_t_f1 += score
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_data_dl:
                # batch_cnt = len(val_datasets) / batch_size
                v_loss, v_acc, v_score, _ = testing(model, images, labels)

                total_v_loss += v_loss
                total_v_acc += v_acc
                total_v_f1 += v_score
        
        train_loss = (total_t_loss/train_batch_cnt).item()
        train_acc = (total_t_acc/train_batch_cnt)
        train_score = (total_t_f1/train_batch_cnt).item()
        val_loss = (total_v_loss/val_batch_cnt).item()
        val_acc = (total_v_acc/val_batch_cnt)
        val_score = (total_v_f1/val_batch_cnt).item()
        
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
        f1_dict['train'].append(train_score)
        f1_dict['val'].append(val_score)

        if epoch%5 == 0:
            print('---check point---')
            print(f"[{epoch:5}/{epochs:5}]  [Train]       loss: {train_loss:.6f}, score: {train_score*100:.4f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]  loss: {val_loss:.6f}, score: {val_score*100:.4f} %\n")
            
        if len(acc_dict['val']) == 1:
            print("saved first")
            print(f"[{epoch:5}/{epochs:5}]  [Train]       loss: {train_loss:.6f}, score: {train_score*100:.4f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]  loss: {val_loss:.6f}, score: {val_score*100:.4f} %\n")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_dict['val'][-1] >= max(acc_dict['val']):
                print(f"[{epoch:5}/{epochs:5}]  saved model")
                print(f"[{epoch:5}/{epochs:5}]  [Train]       loss: {train_loss:.6f}, score: {train_score*100:.4f} %")
                print(f"[{epoch:5}/{epochs:5}]  [Validation]  loss: {val_loss:.6f}, score: {val_score*100:.4f} %\n")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                
        scheduler.step(val_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]       loss: {train_loss:.6f}, score: {train_score*100:.4f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]  loss: {val_loss:.6f}, score: {val_score*100:.4f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Train] accuracy: {train_acc*100:.4f} %, [Validation] accuracy:{val_acc*100:.4f} %")
            break
        
    return loss_dict, acc_dict, f1_dict



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
    

# VGG image preprocessiong
def normalize_image(image):
    
    try:
        image_min = image.min()
        image_max = image.max()
    except:
        image_min = min(image)
        image_max = max(image)
    
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max-image_min+1e-5)
    return image

# show_image
def plot_most_correct(correct, classes, n_images, normalize=True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(25, 20))

    for i in range(rows*cols):
        ax = fig.add_subplot(rows,cols,i+1)
        image, true_label , probs = correct[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        correct_prob, correct_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        correct_label = classes[correct_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f"true label: {true_label} ({true_prob:.3f})\npred label: {correct_label} ({correct_prob:.3f})")
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)



if __name__ == '__main__':
    # utils()
    pass