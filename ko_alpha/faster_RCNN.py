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
import matplotlib.patches as patches
import seaborn as sns
import os
import cv2
import json
import PIL
from PIL import Image

from pycocotools.coco import COCO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# utils
# -----------------------------------------------------------------------
# function name: utils
# prameter : none
# function purpose: show package versions
# function return: none

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 37
torch.manual_seed(RANDOM_STATE)

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
    print(f"random rate: {RANDOM_STATE}\n")
    
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"opencv version: {cv2.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
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
    Show normalized image data in datasets.

    Args:
        datasets (Dataset): The image dataset to visualize.
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 15))

    for idx, data in enumerate(datasets):
        if isinstance(data, tuple):
            # 데이터셋 반환값이 튜플일 경우
            img_data = data[0]  # 첫 번째 요소가 이미지 데이터라고 가정
            target = data[1]  # 두 번째 요소가 라벨이라고 가정
        elif isinstance(data, dict):
            # 데이터셋 반환값이 딕셔너리일 경우
            img_data = data['image']  # 키가 'image'일 경우
            target = data.get('label', 'Unknown')  # 키가 'label'일 경우
        else:
            raise ValueError("Unexpected dataset structure.")

        # Normalize image data
        rotated_images = torch.Tensor(img_data)
        image_min = rotated_images.min()
        image_max = rotated_images.max()
        rotated_images.clamp_(min=image_min, max=image_max)
        rotated_images.add_(-image_min).div_(image_max - image_min + 1e-5)

        # Visualize
        axes[idx // 5][idx % 5].imshow(img_data.permute(1, 2, 0))
        axes[idx // 5][idx % 5].set_title(f"Label: {target}", fontsize=25)
        axes[idx // 5][idx % 5].axis('off')

        if idx == 9:  # Show up to 10 images
            break

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


# custom datasets class
# ---------------------------------------------------------------------------
# class purpose: create datasets
# class name: CustomDatasets
# parameters: image_dir, annotation_file
# attribute field: categry_names, output_file, coco, _create_categories
# class function: init, len, getitem

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 어노테이션 JSON 파일 로드
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 이미지 파일 경로와 해당 레이블 매핑
        self.image_paths = [image['file_name'] for image in self.coco_data['images']]
        self.image_ids = {image['file_name']: image['id'] for image in self.coco_data['images']}
        self.categories = {category['id']: category['name'] for category in self.coco_data['categories']}
        
        # 어노테이션 (이미지에 대한 레이블 및 bbox 정보)
        self.annotations = {image_id: [] for image_id in self.image_ids.values()}
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            self.annotations[image_id].append((category_id, bbox))
        
        # 전체 이미지 경로와 어노테이션을 저장
        self.images = self.image_paths
        self.labels = [self.annotations[self.image_ids[img_path]] for img_path in self.image_paths]

        # 데이터셋 크기
        self.n_samples = len(self.images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])  # 이미지 경로
        image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 열기
        
        image_id = self.image_ids[self.images[idx]]
        annotations = self.labels[idx]  # 해당 이미지의 어노테이션 (category_id, bbox)
        
        # 레이블: 다중 클래스이므로, one-hot encoding 방식으로 레이블 생성
        labels = []
        bboxes = []
        for category_id, bbox in annotations:
            labels.append(category_id - 1)  # 카테고리 ID는 1부터 시작하므로, 0-based로 변경
            bboxes.append(bbox)
        
        # 이미지 전처리
        if self.transform:
            image = self.transform(image)
        
        # 텐서로 반환
        return image, torch.tensor(labels), torch.tensor(bboxes)


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

def predict(model, query_data, support_data, y_data, num_classes):
    # y_data는 이미 long 타입이어야 하므로 unsqueeze를 제거하고 원래 형태로 사용
    if len(y_data.shape) == 1:  # 1D 텐서인 경우
        y_data = y_data.unsqueeze(1)  # (N,) -> (N, 1)
    
    y_data = y_data.to(DEVICE)  # y_data를 모델의 device로 이동
    query_data = query_data.to(DEVICE)  # 쿼리 데이터를 모델의 device로 이동
    support_data = support_data.to(DEVICE)  # 지원 데이터를 모델의 device로 이동

    with torch.no_grad():
        # 예측 수행
        pred = model(query_data, support_data)
        
        # softmax 적용 (다중 클래스 확률 분포)
        pred_probs = F.softmax(pred, dim=1)
        
        # 가장 높은 확률을 가진 클래스 선택 (예측된 클래스)
        pred_labels = torch.argmax(pred_probs, dim=1)
    
    return pred_labels, y_data

def predict_web(model, query_data, support_data):
    query_data = query_data.unsqueeze(0).to(DEVICE)  # 배치 차원 추가
    support_data = support_data.unsqueeze(0).to(DEVICE)  # 배치 차원 추가
    
    with torch.no_grad():
        # 예측 수행
        pred = model(query_data, support_data)
        
        # softmax 적용 (다중 클래스 확률 분포)
        pred_probs = F.softmax(pred, dim=1)
        
        # 가장 높은 확률을 가진 클래스 선택 (예측된 클래스)
        pred_label = torch.argmax(pred_probs, dim=1).item()
    
    return pred_label


# 라벨? 크기가 지멋대로임
def custom_collate_fn(batch):
    query_data, support_data, targets = [], [], []
    
    for data in batch:
        # 예시로 첫 번째 이미지를 query_data로, 나머지를 support_data로 분리
        query_data.append(data[0][0])  # 첫 번째 이미지를 query_data
        support_data.append(data[0][1:])  # 나머지 이미지를 support_data
        targets.append(data[1])  # 라벨
    
    print(f"Targets in batch: {targets}\n")  # 확인용 출력
    
    # 쿼리 데이터와 지원 데이터를 텐서로 변환
    query_data = torch.stack(query_data)
    support_data = torch.stack(support_data)
    
    # 패딩 처리
    padded_targets = pad_sequence([torch.tensor(target) for target in targets], batch_first=True, padding_value=-1)
    
    return query_data, support_data, padded_targets



# model learning
# -------------------------------------------------------------
# - function name: training
# - parameter: dataset, model, epochs, lr, batch_size, patience
# - function return: loss, accuracy score, f1 score, proba
# - optimizer: Adam
# - scheduler: ReduceLROnPlatea, standard: val_loss

def training(model, train_dataset, val_dataset, epochs, num_classes, lr=0.001, batch_size=8, patience=10):
    
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    f1_dict = {'train':[], 'val':[]}
    
    optimizer = optima.Adam(model.parameters(), lr=lr)
    train_data_dl = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    val_data_dl = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate_fn, shuffle=True)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='max')
    
    save_param = './model/custom_fsod_param.pth'
    save_model = './model/custom_fsod_model.pth'

    model.train()
    for epoch in range(1, epochs+1):
        total_t_loss, total_t_acc, total_t_f1 = [], [], []
        total_v_loss, total_v_acc, total_v_f1 = [], [], []
        
        for step, (query_data, support_data, labels) in enumerate(train_data_dl):
            query_data = query_data.to(DEVICE)
            support_data = support_data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            pred = model(query_data, support_data)
            
            loss = nn.CrossEntropyLoss()(pred, labels)
            total_t_loss.append(loss.item())
            
            acc_score = MulticlassAccuracy(num_classes=num_classes)(pred, labels)
            total_t_acc.append(acc_score.item())
            f_score = MulticlassF1Score(num_classes=num_classes)(pred, labels)
            total_t_f1.append(f_score.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()  # 평가 모드
        for step, (query_data, support_data, labels) in enumerate(val_data_dl):
            query_data = query_data.to(DEVICE) 
            support_data = support_data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Validation 데이터에 대한 손실 및 성능 계산
            val_loss, val_acc, val_f1, _ = validation(model, query_data, support_data, labels, num_classes)
            
            total_v_loss.append(val_loss)
            total_v_acc.append(val_acc)
            total_v_f1.append(val_f1)
        
        train_loss = sum(total_t_loss) / len(total_t_loss)
        train_acc = sum(total_t_acc) / len(total_t_acc)
        train_score = sum(total_t_f1) / len(total_t_f1)
        
        val_loss = sum(total_v_loss) / len(total_v_loss)
        val_acc = sum(total_v_acc) / len(total_v_acc)
        val_score = sum(total_v_f1) / len(total_v_f1)
        
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
    
    
def show_mat(mat, title):
    sns.heatmap(mat, annot=True, fmt='.2f', cbar=False, cmap='BuPu')
    plt.title(title + 'confuse matrics')
    plt.show()    


def visualize_annotations(image_dir, annotation_file, image_id=None):
    """
    COCO 어노테이션 파일에 포함된 이미지를 시각화하는 함수.
    
    :param image_dir: 이미지가 저장된 디렉토리
    :param annotation_file: COCO 형식의 어노테이션 파일 경로
    :param image_id: 특정 이미지 ID를 시각화하려면 ID를 전달 (None이면 모든 이미지를 시각화)
    """
    # COCO 어노테이션 파일 로드
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 이미지와 어노테이션 매핑
    images = {image['id']: image for image in coco_data['images']}
    annotations = coco_data['annotations']
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    
    # 특정 이미지 ID가 있으면 해당 이미지만 필터링
    if image_id is not None:
        images = {image_id: images[image_id]}
    
    # 이미지별로 시각화
    for img_id, img_info in images.items():
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"이미지 파일 '{img_path}'을 찾을 수 없습니다.\n")
            continue
        
        # 이미지 열기
        image = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # 해당 이미지의 어노테이션 필터링
        img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
        
        # 바운딩 박스 그리기
        for ann in img_annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_name = categories[ann['category_id']]
            
            # 바운딩 박스 추가
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # 클래스 이름 추가
            ax.text(
                bbox[0], bbox[1] - 5, category_name,
                color='red', fontsize=12, backgroundcolor='white'
            )
        
        # 결과 표시
        plt.title(f"Image ID: {img_id}")
        plt.show()