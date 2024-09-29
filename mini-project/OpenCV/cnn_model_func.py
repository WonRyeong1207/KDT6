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
# ----------------------------------------------------------------
# function name: image_transformer
# parameter: kind(train, test)
# function purpose: select image data transformer
# function return: transfromer

def image_transformer(kind=['train', 'test']):
    """
    select image data transformer

    Args:
        kind (str, optional): choose transmode in [train, test]. Defaults to 'train'.

    Returns:
        v2.Compose instance: image transformer
    """
    if kind == 'train':
        transformer = v2.Compose([
            v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomResizedCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transformer = v2.Compose([
            v2.Resize(size=(256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transformer


# show images in datasets
# -------------------------------------------------------------------
# function name: show_image_dataset
# 
def show_image_dataset(datasets):
    name = datasets.classes

    fig, axes = plt.subplots(2, 5, figsize=(24, 15))

    for idx, (img_data, tagest) in enumerate(datasets):

        rotated_images = torch.Tensor(img_data)
        # normalization code
        image_min = rotated_images.min()
        image_max = rotated_images.max()
        rotated_images.clamp_(min=image_min, max=image_max)
        rotated_images.add_(-image_min).div_(image_max - image_min + 1e-5)
        
        axes[idx//5][idx%5].imshow(img_data.permute(1, 2 ,0))
        axes[idx//5][idx%5].set_title(name[tagest], fontsize=25)
        axes[idx//5][idx%5].axis('off')
        
        if idx == 9: break
        
    plt.tight_layout()
    # plt.axis('off')
    plt.show()
    

# model custom
class CustomVgg16Model(nn.Module):
    def __init__(self):
        super(CustomVgg16Model, self).__init__()    # 이러면 전의학습 들고오는가봄
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
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        y = self.features(x)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        y = F.sigmoid(self.custom_layer(y))
        
        return y


def predict(model, images, labels):
    
    with torch.no_grad():
        pred = model(images)
        pred_labels = (pred > 0.5).int()
        pred_labels = [LABEL_TRANSLATE[int(label)] for label in pred_labels.flatten()]
        real_labels = [LABEL_TRANSLATE[int(label)] for label in labels.flatten()]
    
    return pred_labels, real_labels

def predict_web(model, images):
    # model.eval()
    with torch.no_grad():
        pred = model(images)
        pred_labels = (pred > 0.5).int()
        pred_labels = LABEL_TRANSLATE[int(pred_labels.flatten())]
        # real_labels = [LABEL_TRANSLATE[int(label)] for label in labels.flatten()]
    
    return pred_labels


def testing(model, test_images, test_labels):
    test_labels = test_labels.unsqueeze(1).float()  # 차원이 맞지 않기 때문에 차원 추가
    
    with torch.no_grad():
        pred = model(test_images)
        loss = nn.BCELoss()(pred, test_labels)
        acc = BinaryAccuracy()(pred, test_labels)
        f1 = BinaryF1Score()(pred, test_labels)
        mat = BinaryConfusionMatrix()(pred, test_labels)
        
    return loss, acc, f1, mat


def training(model, train_datasets, val_datasets, epochs=100, lr=0.001, batch_size=64, patience=5):
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    f1_dict = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    train_data_dl = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_data_dl = DataLoader(val_datasets, batch_size=10, shuffle=True)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    
    save_param = './model/CNN_bc_custom_clf_params.pth'
    save_model = './model/CNN_bc_custom_clf_model.pth'
    
    train_batch_cnt = len(train_datasets) / batch_size
    val_batch_cnt = len(val_datasets) / 10
    
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = 0, 0, 0
        total_v_loss, total_v_acc, total_v_f1 = 0, 0, 0
        
        model.train()
        for images, labels in train_data_dl:
            # batch_cnt = len(train_datasets) / batch_size
            labels = labels.unsqueeze(1).float()

            pred = model(images)
            
            loss = nn.BCELoss()(pred, labels)
            total_t_loss += loss
            
            acc = BinaryAccuracy()(pred, labels)
            total_t_acc += acc
            
            score = BinaryF1Score()(pred, labels)
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
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(r2['train'], label=f"train score max: {max(r2['train'])*100:.2f} %", color='#00007F')
    ax2.plot(r2['val'], label=f"validation score max: {max(r2['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} iris CNN binary classification', fontsize=15)
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()
    

def show_predict_image(model, data_loader, num):
    cnt = 0
    for image, label in data_loader:
        if cnt == num:
            break
        
        image = image
        label = label
        cnt += 1
    
    rotated_images = torch.Tensor(image[num])
    # normalization code
    image_min = rotated_images.min()
    image_max = rotated_images.max()
    rotated_images.clamp_(min=image_min, max=image_max)
    rotated_images.add_(-image_min).div_(image_max - image_min + 1e-5)
    
    pred_label, real_label = predict(model, image, label)
    
    plt.imshow(rotated_images.permute(1, 2, 0).numpy())
    plt.title(f"predict: {pred_label[num]}\nreal: {real_label[num]}", color='green' if pred_label[num] == real_label[num] else 'red')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    

