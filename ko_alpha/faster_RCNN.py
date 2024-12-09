"""
for few-shot object detection model by faster-RCNN file
- frame work : python, pytorch, torchvision, flask
---
- base datasets : MS-COCO 2017
- custom datasets = few-shot datasets
- learning method : supervised learning, multi classification, transfer learning, meta learning, few-shot learning
- learning algorithm : CNN
- transfer learning model : faster RCNN (backbone model : pre-trained resnet50)
---
actually, I don't understand all.

"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import json
import PIL
from PIL import Image

from pycocotools.coco import COCO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optima
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.models as models
from torchvision.transforms import v2

import torchinfo
from torchinfo import summary

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score


# utils
# -----------------------------------------------------------------------
# function name: utils
# prameter : none
# function purpose: show package versions
# function return: none

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_RATE = 37
torch.manual_seed(RANDOM_RATE)

def utils():
    """
    show & check utils
    """
    print("----- Notice -----")
    print(f"device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"cuda num: {torch.cuda.device_count()}")
        print(f"cude device name: {torch.cuda.get_device_name(0)}")
        print(f"cudnn version: {torch.backends.cudnn.version()}")
    print(f"random rate: {RANDOM_RATE}\n")
    
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"opencv version: {cv2.__version__}")
    print(f"PIL version: {PIL.__version__}\n")
    
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"torchinfo version: {torchinfo.__version__}")
    print(f"torchmetrics version: {torchmetrics.__version__}")
    

# image transform function
# -------------------------------------------------------------------
# fonction name: image_transformer
# parameter: kind(train, test)
# function purpose: select image data transformer
# function return: transformer

def image_transformer(kind='train'):
    """
    select image data transformer
    ---
    ---
    Args:
        kind (str, optional): choose transmode in [train, test]. Defaults to 'train'.

    Raises:
        ValueError: kind not in [train, test]

    Returns:
        v2.Compose instance: image transformer
    """
    
    if kind not in ['train', 'test']:
        raise ValueError("Invalid kind. Allowed values are 'train' or 'test'.")
    
    if kind == 'train':
        transformer = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop(224),
            v2.ToTensor(),  # 이미지를 텐서로 변환하고 [0, 1]로 스케일링
            v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        transformer = v2.Compose([
            v2.Resize(size=256, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
    return transformer


# show images in datasets
# -------------------------------------------------------------------
# function name: show_image_dataset
# parameter: datasets
# function purpose: show normalize image data
# function return: none

def show_image_dataset(datasets):
    """
    show noralized image data in datasets
    ---
    ---
    Args:
        datasets (datssets): want to see image datasets
    """
    
    name = datasets.classes

    fig, axes = plt.subplots(2, 5, figsize=(24, 15))

    for idx, (img_data, tagest) in enumerate(datasets):

        rotated_images = torch.Tensor(img_data)
        # normalization code
        image_min = rotated_images.min()
        image_max = rotated_images.max()
        rotated_images.clamp_(min=image_min, max=image_max)
        rotated_images.add_(-image_min).div_(image_max - image_min + 1e-5)
        
        # image data를 보고 결정
        axes[idx//5][idx%5].imshow(img_data.permute(1, 2 ,0))
        axes[idx//5][idx%5].set_title(name[tagest], fontsize=25)
        axes[idx//5][idx%5].axis('off')
        
        if idx == 9: break
        
    plt.tight_layout()
    plt.show()
    
    
# custom faster rcnn model class
# -----------------------------------------------------------------------
# class purpose: classification object, object detection
# class name: CustomFasterRCNNModel
# parameters: num_classes
# attribute field: self.faster_rcnn, num_classes
# class function: create model structure, forward learning model
# class structure
# - transform (이건 없앨 수도 있음.)
# - backbone
# - RPN
# - roi_heads

class CustomFasterRCNNModel(nn.Module):
    # 기본 초기화 모델
    def __init__(self, num_classes, min_k=1, max_k=10):
        super(CustomFasterRCNNModel,self).__init__()
        # 기존의 model 정보를 그대로 저장
        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # 모델 선언 이후 새로 입력할 수고를 덜기 위함.
        self.num_classes = num_classes # 기본이 91 + novel class
        self.min_k, self.max_k = min_k, max_k

        
        # self.transform = self.faster_rcnn.transform
        # self.backbone = self.faster_rcnn.backbone
        # self.rpn = self.faster_rcnn.rpn
        
        # self.roi_heads = self.faster_rcnn.roi_headss
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
        
    # knn 유사도 계산하는 함수
    # query_image: 내가 인식하고 싶은 이미지
    # support_image: query_image를 보조하는 이미지
    def compute_knn_similarity(self, query_features, support_features, k):
        # # 특징 벡터들의 유사도 계산 (유클리드 거리)
        # distances = torch.cdist(query_features.flatten(1), support_features.flatten(1))  # (N_query, N_support)
        
        # # K개의 최근접 이웃을 찾음
        # _, knn_indices = torch.topk(distances, k, dim=1, largest=False, sorted=False)
        
        # # K개의 이웃의 유사도(거리) 평균을 계산하여 반환
        # knn_similarities = torch.mean(distances.gather(1, knn_indices), dim=1)
        
        # return knn_similarities
        
        # OrderedDict에서 특정 특징 맵 선택
        query_features = query_features['0']  # 예: '0' 키를 선택 (다른 키 필요시 변경)
        support_features = support_features['0']

        # 특징 벡터들의 유사도 계산 (유클리드 거리)
        distances = torch.cdist(query_features.flatten(1), support_features.flatten(1))  # (N_query, N_support)

        # K개의 최근접 이웃을 찾음
        _, knn_indices = torch.topk(distances, k, dim=1, largest=False, sorted=False)

        return knn_indices
        
    # function of finding optima k
    def optimize_k(self, query_features, support_features):
        # # 다양한 k 값에 대해 성능을 평가할 수 있는 메커니즘 필요
        # best_k = self.min_k
        # best_similarity = None
        # best_loss = float('inf')

        # # 후보 k값들에 대해 성능을 평가 (예: loss 값)
        # for k in range(self.min_k, self.max_k + 1):
        #     knn_similarities = self.compute_knn_similarity(query_features, support_features, k)
        #     # 여기서는 임시로 유사도의 평균을 사용하여 loss를 계산 (더 정교한 평가 필요)
        #     loss = knn_similarities.mean().item()
            
        #     if loss < best_loss:
        #         best_loss = loss
        #         best_k = k

        # return best_k
        
        best_k = None
        min_loss = float('inf')

        # 후보 k값들에 대해 성능을 평가 (예: loss 값)
        for k in range(self.min_k, self.max_k + 1):
            knn_similarities = self.compute_knn_similarity(query_features, support_features, k)
            # 여기서는 임시로 유사도의 평균을 사용하여 loss를 계산 (더 정교한 평가 필요)
            loss = knn_similarities.float().mean().item()

            if loss < min_loss:
                min_loss = loss
                best_k = k

        return best_k
    
    # knn 유사도를 기반으로 RoI를 필터링하거나 추가적인 FSOD 논리를 수행하는 함수
    def custom_roi_heads(self, query_features, support_features, targets=None):
         # '0' 키로 Tensor 추출
        query_tensor = query_features.get('0')
        support_tensor = support_features.get('0')
        
        if query_tensor is None or support_tensor is None:
            raise ValueError("Expected key '0' not found in OrderedDict.")
        
        print(f"query_tensor shape: {query_tensor.shape}")
        print(f"support_tensor shape: {support_tensor.shape}\n")
        
        # Tensor가 있을 경우 flatten 처리
        query_tensor = query_tensor.flatten(1) if isinstance(query_tensor, torch.Tensor) else query_tensor
        support_tensor = support_tensor.flatten(1) if isinstance(support_tensor, torch.Tensor) else support_tensor

        # 최적의 K값을 자동으로 계산
        best_k = self.optimize_k(query_features, support_features)
        
        # 최적의 K값으로 유사도 계산
        knn_similarities = self.compute_knn_similarity(query_features, support_features, best_k)
        
        # 유사도가 높은 RoI만 필터링
        threshold = knn_similarities.float().mean()  # 평균 유사도를 임계값으로 사용
        high_similarity_indices = knn_similarities < threshold

        print(f"knn_similarities shape: {knn_similarities.shape}")  # 유사도의 크기
        print(f"high_similarity_indices shape: {high_similarity_indices.shape}\n")  # 필터링된 인덱스 크기
        
        # high_similarity_indices의 크기를 query_tensor와 맞추기 위해 확장
        high_similarity_indices_expanded = high_similarity_indices.expand(-1, query_tensor.size(1))  # (16, 256)

        # 유사도가 높은 부분만 필터링하여 RoI 처리
        refined_features = query_tensor[high_similarity_indices_expanded]
        
        # 이미지 크기 계산 (배치 크기와 이미지 크기)
        image_shapes = query_tensor.shape[-2:]  # (height, width)

        # 일반적인 RoI 헤드를 통해 결과 얻기
        output = self.faster_rcnn.roi_heads(refined_features, targets, image_shapes)
        
        return output

    # 초기 학습 함수
    # def forward(self, images, targets=None):
    #     # output = self.transform(inputs)
    #     # output = self.backbone(output)
    #     # output = self.rpn(output)
    #     # y = self.roi_heads(output)
        
    #     output = self.faster_rcnn(images, targets)
        
    #     return output
    
    # model learning function
    def forward(self, query_images, support_images, targets=None):
        # Backbone에서 Query와 Support 특징 추출
        query_features = self.faster_rcnn.backbone(query_images)
        support_features = self.faster_rcnn.backbone(support_images)
        # image를 잘못 나누면 필요없..
        
        # KNN 유사도를 기반으로 RoI를 필터링하거나 추가적인 FSOD 논리 수행
        refined_results = self.custom_roi_heads(query_features, support_features, targets)
        
        return refined_results
    

# custom datasets class
# ----------------------------------------------------------------------------
# class purpose: create data format by coco annotaion
# class name: CreateNovelClass
# parameters: category_names, output_file
# attribute field: categry_names, output_file, coco, _create_categories
# class function: init, _create_categories, add_image, save

class CreateNovelClass:
    def __init__(self, category_name_list, output_file_path):
        # param category_names: 새로운 클래스의 이름들
        # param output_file: 저장할 COCO 형식의 어노테이션 파일 경로
        
        self.category_names = category_name_list
        self.output_file = output_file_path

        # COCO 기본 구조 초기화
        self.coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 카테고리 추가
        self._create_categories()

    def _create_categories(self):
        # 카테고리 정보를 COCO 형식에 맞게 추가
        self.coco["categories"] = [{"id": i + 1, "name": name} for i, name in enumerate(self.category_names)]

    def add_image(self, img_id, img_path, bboxes, labels):
        # 새로운 이미지와 어노테이션을 추가.
        # param img_id: 이미지의 고유 ID
        # param img_path: 이미지 파일 경로
        # param bboxes: 이미지에 대한 bounding box 정보 [[x, y, width, height], ...]
        # param labels: 각 bounding box에 대한 label (클래스 ID) 리스트
        
        # 이미지 정보
        image_info = {
            "id": img_id,
            "file_name": img_path,
            "width": 640,  # 예시로 이미지의 크기를 지정 (실제 이미지 크기에 맞게 수정 가능)
            "height": 480  # 예시로 이미지의 크기를 지정 (실제 이미지 크기에 맞게 수정 가능)
        }
        self.coco["images"].append(image_info)

        # 어노테이션 정보
        annotation_id = len(self.coco["annotations"]) + 1
        for bbox, label in zip(bboxes, labels):
            annotation_info = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": label,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],  # width * height
                "iscrowd": 0  # 객체가 군집이 아닌 경우
            }
            self.coco["annotations"].append(annotation_info)
            annotation_id += 1

    def save(self):
        # COCO 형식의 어노테이션 파일 저장
        with open(self.output_file, 'w') as f:
            json.dump(self.coco, f)


# 이거는 이미 존재하는 coco annotation의 format을 그대로 사용하는 코드
class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # 이미지 로드
        image_path = os.path.join(self.image_dir, self.coco.imgs[img_id]['file_name'])
        image = Image.open(image_path).convert("RGB")

        # annotation 처리
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)




# validation & test function
# -----------------------------------------------------
# - function name: testing
# - parameter: model, x_data, y_data
# - function return: loss, score_accuracy, score_f1score
# -----------------------------------------------------
# must not update weight & bais

def validation(model, query_data, support_data, y_data, num_classes):
    # query_data는 (N, 4) 형태의 바운딩 박스를 포함한다고 가정
    # y_data는 (N,) 형태의 레이블 정보
    targets = [{
        "boxes": query_data,  # 바운딩 박스 정보
        "labels": y_data      # 객체의 레이블 정보
    }]
    
    with torch.no_grad():
        pred = model(query_data, support_data, targets)  # targets도 전달
        
        # 손실 함수 및 정확도 계산
        loss = nn.CrossEntropyLoss()(pred, y_data)
        acc_score = MulticlassAccuracy(num_classes=num_classes)(pred, y_data)
        f1_score = MulticlassF1Score(num_classes=num_classes)(pred, y_data)
        mat = MulticlassConfusionMatrix(num_classes=num_classes)(pred, y_data)
        
    return loss, acc_score, f1_score, mat

def testing(model, query_data, support_data, y_data, num_classes):
     # y_data는 이미 long 타입이어야 하므로 unsqueeze를 제거하고 원래 형태로 사용
    targets = [{
        "boxes": query_data,  # 바운딩 박스 정보
        "labels": y_data      # 객체의 레이블 정보
    }]
    
    with torch.no_grad():
        pred = model(query_data, support_data, targets)  # targets도 전달
        
        # 손실 함수 및 정확도 계산
        loss = nn.CrossEntropyLoss()(pred, y_data)
        acc_score = MulticlassAccuracy(num_classes=num_classes)(pred, y_data)
        f1_score = MulticlassF1Score(num_classes=num_classes)(pred, y_data)
        mat = MulticlassConfusionMatrix(num_classes=num_classes)(pred, y_data)
        
    return loss, acc_score, f1_score, mat


# predict function
# -----------------------------------------------------
# - function name: predict
# - parameter: model, X_data
# - function return: loss, score_accuracy, score_f1score, proba
# -----------------------------------------------------
# must not update weight & bais

def predict_web(model, X_data, type_='others'):
    X_data = X_data.unsqueeze(1)
    with torch.no_grad():
        if type_=='me':
            pred = model(X_data)
            # pred = torch.max(pred)
            # pred = torch.argmax(pred)
        elif type_=='others':
            pred = model(X_data)
            pred = torch.sigmoid(pred)
            # pred = torch.max(pred)
            # pred = torch.argmax(pred)
        # pred_label = torch.argmax((pred > 0.5).int())
        # pred_label = int(pred_label.flatten())
        # if pred_label > 1:
        #     pred_label = 0
        # pred_label = lable_translate[pred_label]
        pred = torch.max(pred)
        
    return pred.item()

def predict(model, x_data, y_data):
    # 텐서 하나
    # y_data의 차원을 확인하고 unsqueeze가 필요한지 확인
    if len(y_data.shape) == 1:  # 1D 텐서인 경우
        y_data = y_data.unsqueeze(1)  # (N,) -> (N, 1)
    
    y_data = y_data.float()  # float형으로 변환
    x_data = x_data.unsqueeze(1)
    with torch.no_grad():
        # 예측 수행
        pred = model(x_data)
        # 0.5 이상이면 1, 미만이면 0으로 변환
        pred_labels = torch.argmax(pred)
        # pred_labels = [LABEL_TRANSLATE[int(label)] for label in pred_labels.flatten()]
        # real_labels = [LABEL_TRANSLATE[int(label)] for label in y_data.flatten()]

    return pred_labels, y_data


# model learning
# -------------------------------------------------------------
# - function name: training
# - parameter: dataset, model, epochs, lr, batch_size, patience
# - function return: loss, accuracy score, f1 score, proba
# - optimizer: Adam
# - scheduler: ReduceLROnPlatea, standard: val_loss

def training(model, train_dataset, val_dataset, epochs, lr=0.001, batch_size=32, patience=10):
    
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    f1_dict = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    train_data_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_dl = DataLoader(val_dataset, batch_size=100, shuffle=True)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='max')
    
    save_param = './model/custom_fsod_param.pth'
    save_model = './model/custom_fsod_model.pth'
    
    train_batch_cnt = len(train_dataset) // batch_size
    val_batch_cnt = len(val_dataset) // 100

    model.train()
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = [], [], []
        total_v_loss, total_v_acc, total_v_f1 = [], [], []
        
        for step, (input_ids, labels) in enumerate(train_data_dl):
            # batch_cnt = dataset.n_rows / batch_size
            labels = labels.unsqueeze(1)
            
            pred = model(input_ids)
            
            loss = nn.CrossEntropyLoss()(pred, labels)
            total_t_loss.append(loss)
            
            a_score = MulticlassAccuracy()(pred, labels)
            total_t_acc.append(a_score)
            f_score = MulticlassF1Score()(pred, labels)
            total_t_f1.append(f_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for step, (input_ids, labels) in enumerate(val_data_dl):
            labels = labels.unsqueeze(1)
            val_loss, val_acc, val_score = validation(model, input_ids, labels)
            
            total_v_loss.append(val_loss)
            total_v_acc.append(val_acc)
            total_v_f1.append(val_score)
        
        if len(total_t_loss) == train_batch_cnt:
            train_loss = (sum(total_t_loss)/train_batch_cnt).item()
            train_acc = (sum(total_t_acc)/train_batch_cnt)
            train_score = (sum(total_t_f1)/train_batch_cnt).item()
            val_loss = (sum(total_v_loss)/val_batch_cnt).item()
            val_acc = (sum(total_v_acc)/val_batch_cnt)
            val_score = (sum(total_v_f1)/val_batch_cnt).item()
        else:
            train_loss = (sum(total_t_loss)/len(total_t_loss)).item()
            train_acc = (sum(total_t_acc)/len(total_t_acc))
            train_score = (sum(total_t_f1)/len(total_t_f1)).item()
            val_loss = (sum(total_v_loss)/len(total_v_loss)).item()
            val_acc = (sum(total_v_acc)/len(total_v_acc))
            val_score = (sum(total_v_f1)/len(total_v_f1)).item()
        
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
        f1_dict['train'].append(train_score)
        f1_dict['val'].append(val_score)
        
        if epoch%5 == 0:
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score*100:.6f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_score*100:.6f} %\n")
        
        if len(acc_dict['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_dict['val'][-1] >= max(acc_dict['val']):
                print(f"[{epoch:5}/{epochs:5}] saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                
        scheduler.step(val_acc)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_score*100:.6f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_score*100:.6f} %\n")
            break
        
    return loss_dict, acc_dict, f1_dict
    

# 그림 그리는 함수
def draw_two_plot(loss, score, title, type_='FSOD'):
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(score['train'], label=f"train score max: {max(score['train'])*100:.2f} %", color='#00007F')
    ax2.plot(score['val'], label=f"validation score max: {max(score['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} {type_} binary classification', fontsize=15)
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()
    
    
