# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import torch      # AI 모델 관련
import sys, codecs # 인코딩 관련
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning')
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model')
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import cnn_multi_class_func as cnn # 내가 만든거
from torchvision import transforms
from PIL import Image

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            <!-- form action 경로를 'webDetectImage_cnn.py'로 변경 -->
            <form enctype="multipart/form-data" action="/cgi-bin/webDetectImage_cnn.py" method="POST">
                <input type="file" name="imageInput" accept="image/*">
                <input type="submit" value="Classify">
            </form>
            <div id="result">{msg}</div>
        </body>
        </html>
         """)



# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detect_cnn_image
# 재 료 : 사용자 입력 이미지파일
# 결 과 : 동물 종

def detect_cnn_image(image_file):
    
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    
    # 단일 이미지 파일 처리
    image_pil = Image.open(image_file)
    image_transformed = img_transforms(image_pil).unsqueeze(0)  # 배치 차원 추가
    # image_transformed = img_transforms(image_pil)
    
    image_transformed = image_transformed.to(DEVICE)
    # 모델에 입력 및 결과 반환
    result = cnn.predict_web(vgg_model, image_transformed)
    
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
    # pklfile = os.path.abspath('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model\\dct_multi_clf.pth')
    pklfile_2 = os.path.abspath('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\DeepLearning\\model\\gvv16_model_epoch100.pth')

if not os.path.exists(pklfile_2):
    raise FileNotFoundError(f"Model file not found at: {pklfile_2}")

vgg_model = cnn.VGG16Model()
vgg_model = torch.load(pklfile_2, weights_only=False, map_location=torch.device('cpu'))

# (3) WEB 사용자 입력 데이터 처리
form = cgi.FieldStorage()
image_file = 'abc'

# 이미지 파일 읽기
if "imageInput" in form:
    image_file = form['imageInput'].file  # 업로드된 파일 객체
    result_cnn = detect_cnn_image(image_file)
    msg = f"Predicted Animal: {result_cnn}"
else:
    msg = "No image uploaded."

# 결과 출력
showHTML(msg)