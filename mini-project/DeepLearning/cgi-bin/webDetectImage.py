# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import torch      # AI 모델 관련
import sys, codecs # 인코딩 관련
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import dct_model_class_func as work    # 내가 만든거
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(text, msg):
    print("Content-Type: text/html; charset=utf-8")
    print(f"""
        <!DOCTYPE html>
        <html lang="ko">

        <head>
            <meta charset="UTF-8">
            <title>Image Classification</title>
        </head>

        <body>
            <h1>Image Classifier</h1>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="classifyImage()">Classifier</button>
            <div id="result"></div>
            <img id="uploadedImage" style="max-width: 300px; margin-top: 20px;">
        </body>

        </html>
         """)

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detect_dct_image
# 재 료 : 사용자 입력 이미지파일
# 결 과 : 동물 종

def detect_dct_image(text):
    # dct 변환
    image = cv2.imread(text, cv2.IMREAD_COLOR)
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
    image_ts = torch.FloatTensor(image_df.values)
    
    # 판별요청 & 결과 반환
    result = work.predict(dct_Model, image_ts)
    
    return result

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    pklfile = os.path.dirname(__file__)+ '/model/dct_multi_clf.pkl' # 웹상에서는 절대경로만
else:
    pklfile = '/model/dct_multi_clf.pkl'
    
dct_Model = torch.load(pklfile)

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태크 속 데이터 가져오기
text = form.getvalue("text", default="")
#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기
msg =""
if text != "":
    resultLang = detect_dct_image(text)
    msg = f"{resultLang}"

# (4) 사용자에게 WEB 화면 제공
showHTML(text,msg)
