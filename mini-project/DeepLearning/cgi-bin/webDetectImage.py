# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import torch      # AI 모델 관련
import sys, codecs # 인코딩 관련
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning')
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model')
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import dct_model_class_func as work    # 내가 만든거
import cnn_multi_class_func as cnn
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(msg):
    print("Content-Type: text/html; charset=utf-8")
    print("Cache-Control: no-cache, no-store, must-revalidate")  # 캐시 방지
    print("Pragma: no-cache")  # HTTP 1.0 캐시 방지
    print("Expires: 0")  # 구형 브라우저 캐시 방지
    print(f"""
        <!DOCTYPE html>
        <html lang="ko">

        <head>
            <meta charset="UTF-8">
            <title>Image Classification</title>
        </head>

        <body>
            <h1>Image Classifier</h1>
            <!-- form action 경로를 'webDetectImage.py'로 변경 -->
            <form enctype="multipart/form-data" action="/cgi-bin/webDetectImage.py" method="POST">
                <input type="file" name="imageInput" accept="image/*">
                <input type="submit" value="Classify">
            </form>
            <div id="result">{msg}</div>
        </body>
        </html>
         """)


# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detect_dct_image
# 재 료 : 사용자 입력 이미지파일
# 결 과 : 동물 종

def detect_dct_image(image_file):
    # dct 변환
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_resize = cv2.resize(gray, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    imsize = gray_resize.shape
    gray2 = np.float32(gray)
    dct = np.zeros(imsize, dtype=float)
    
    blk_slc = 8

    for i in range(0, imsize[0], blk_slc):
        for j in range(0, imsize[1], blk_slc):
            dct[i:(i+blk_slc), j:(j+blk_slc)] = cv2.dct(gray2[i:(i+blk_slc), j:(j+blk_slc)])
            dct.astype(np.uint8)
    
    image_df = pd.DataFrame(dct.reshape(-1)).T
    
    data_df = pd.read_csv('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\dct_image.csv')
    feature_df = data_df[data_df.columns[:-1]]
    label_sr = data_df[data_df.columns[-1]]
    
    scaler = StandardScaler()
    scaler.fit(feature_df, label_sr)
    image_df = scaler.transform(image_df)
    
    image_ts = torch.FloatTensor(image_df)
    
    # 판별요청 & 결과 반환
    result = work.predict(dct_Model, image_ts)
    
    return result

# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detect_cnn_image
# 재 료 : 사용자 입력 이미지파일
# 결 과 : 동물 종

def detect_cnn_image(image_file):
    
    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    
    img_datasets = ImageFolder(root=image_file.read(), transform=img_transforms)
    test_loader = DataLoader(img_datasets, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    # 이미지 파일을 읽고 변환
    # image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 변환
    # image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    # image_pil = Image.fromarray(image)  # PIL 이미지로 변환
    # image_transformed = img_transforms(image_pil).unsqueeze(0)  # 변환 후 배치 차원 추가

    # 판별 요청 & 결과 반환
    result, _ = cnn.predict(vgg_model, test_loader)
    
    return result


# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
#     pklfile = os.path.join(os.path.dirname(__file__), 'dct_multi_clf.pth')  # 올바른 경로 지정
# else:
#     pklfile = 'C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model\\dct_multi_clf.pth'  # 경로 수정
# pklfile = os.path.join(os.path.dirname(__file__), 'model', 'dct_multi_clf.pth')
    pklfile = os.path.abspath('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model\\dct_multi_clf.pth')
    pklfile_2 = os.path.abspath('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model\\gvv16_model_epoch100.pth')

if not os.path.exists(pklfile):
    raise FileNotFoundError(f"Model file not found at: {pklfile}")

dct_clf_model = work.DctMCModel()
dct_Model = torch.load(pklfile, weights_only=False)
vgg_model = cnn.VGG16Model()
vgg_model = torch.load(pklfile_2, weights_only=False, map_location=torch.device('cpu'))

# (3) WEB 사용자 입력 데이터 처리
form = cgi.FieldStorage()
image_file = 'abc'

# 이미지 파일 읽기
if "imageInput" in form:
    image_file = form['imageInput'].file  # 업로드된 파일 객체
    result_animal = detect_dct_image(image_file)
    result_cnn = detect_cnn_image(image_file)
    msg = f"Predicted Animal: {result_animal}, {result_cnn}"
else:
    msg = "No image uploaded."

# 결과 출력
showHTML(msg)