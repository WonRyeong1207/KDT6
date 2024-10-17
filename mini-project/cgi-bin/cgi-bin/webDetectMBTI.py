# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import torch      # AI 모델 관련
import sys, codecs # 인코딩 관련
sys.path.append(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin')
sys.path.append(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\model')
sys.path.append(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\scaler')
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import sklearn
import pandas as pd
import numpy as np
import pickle

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(msg, inputs):
    print("Content-Type: text/html; charset=utf-8")
    print("Cache-Control: no-cache, no-store, must-revalidate")  # 캐시 방지
    print("Pragma: no-cache")  # HTTP 1.0 캐시 방지
    print("Expires: 0")  # 구형 브라우저 캐시 방지
    print(f"""
        <!DOCTYPE html>
        <html lang="ko">
         <head>
          <meta charset="UTF-8">
          <title>MBTI Classification</title>
          <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 800px;
                margin: 20px auto;
                background-color: #fff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                border-radius: 8px;
            }}
            .input-group {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .input-group label {{
                flex: 1;
                font-size: 16px;
                padding-right: 10px;
            }}
            .input-group input {{
                flex: 2;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .submit-btn {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                border-radius: 4px;
            }}
            .submit-btn:hover {{
                background-color: #45a049;
            }}
            .answer-section {{
                margin-top: 20px;
                padding: 20px;
                background-color: #f1f1f1;
                border-radius: 4px;
            }}
          </style>
         </head>
        
         <body>
          <div class="container">
            <h2>트위터 기반 MBTI classification</h2>
            <form method="post">
              <div class="input-group">
                <label for="total_retweet_count">Retweet Count:</label>
                <input type="number" name="total_retweet_count" id="total_retweet_count" placeholder="Retweet Count" value="{inputs['total_retweet_count']}">
              </div>
              <div class="input-group">
                <label for="total_favorite_count">Favorite Count:</label>
                <input type="number" name="total_favorite_count" id="total_favorite_count" placeholder="Favorite Count" value="{inputs['total_favorite_count']}">
              </div>
              <div class="input-group">
                <label for="total_hashtag_count">Hashtag Count:</label>
                <input type="number" name="total_hashtag_count" id="total_hashtag_count" placeholder="Hashtag Count" value="{inputs['total_hashtag_count']}">
              </div>
              <div class="input-group">
                <label for="total_url_count">URL Count:</label>
                <input type="number" name="total_url_count" id="total_url_count" placeholder="URL Count" value="{inputs['total_url_count']}">
              </div>
              <div class="input-group">
                <label for="total_mentions_count">Mentions Count:</label>
                <input type="number" name="total_mentions_count" id="total_mentions_count" placeholder="Mentions Count" value="{inputs['total_mentions_count']}">
              </div>
              <div class="input-group">
                <label for="total_media_count">Media Count:</label>
                <input type="number" name="total_media_count" id="total_media_count" placeholder="Media Count" value="{inputs['total_media_count']}">
              </div>
              <div class="input-group">
                <label for="number_of_tweets_scraped">Tweets Scraped:</label>
                <input type="number" name="number_of_tweets_scraped" id="number_of_tweets_scraped" placeholder="Number of Tweets Scraped" value="{inputs['number_of_tweets_scraped']}">
              </div>
              <button type="submit" class="submit-btn">결과</button>
            </form>
            
            <div class="answer-section">
              <h3>MBTI 결과:</h3>
              <p>{msg} 입니다!</p>
            </div>
          </div>
         </body>
        </html>""")



# file load
def load(file_path):
    with open(file_path, 'rb') as f:
        caryy = pickle.load(f)
    return caryy

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    # model load
    pklfile_ei_model = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\model\ei_model'
    pklfile_ns_model = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\model\ns_model'
    pklfile_tf_model = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\model\tf_model'
    pklfile_pj_model = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\model\pj_model'
    # scaler load
    pklfile_ei_scaler = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\scaler\ei_sd_scaler'
    pklfile_ns_scaler = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\scaler\ns_sd_scaler'
    pklfile_tf_scaler = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\scaler\tf_sd_scaler'
    pklfile_pj_scaler = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\cgi-bin\scaler\pj_sd_scaler'

# model
ei_model = load(pklfile_ei_model)
ns_model = load(pklfile_ns_model)
tf_model = load(pklfile_tf_model)
pj_model = load(pklfile_pj_model)

# scaler
ei_scaler = load(pklfile_ei_scaler)
ns_scaler = load(pklfile_ns_scaler)
tf_scaler = load(pklfile_tf_scaler)
pj_scaler = load(pklfile_pj_scaler)

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태크 속 데이터 가져오기

inputs = {
    "total_retweet_count": form.getvalue("total_retweet_count", "0"),
    "total_favorite_count": form.getvalue("total_favorite_count", "0"),
    "total_hashtag_count": form.getvalue("total_hashtag_count", "0"),
    "total_url_count": form.getvalue("total_url_count", "0"),
    "total_mentions_count": form.getvalue("total_mentions_count", "0"),
    "total_media_count": form.getvalue("total_media_count", "0"),
    "number_of_tweets_scraped": form.getvalue("number_of_tweets_scraped", "0"),
}

# 각 값을 정수로 변환
total_retweet_count = int(inputs['total_retweet_count'])
total_favorite_count = int(inputs['total_favorite_count'])
total_hashtag_count = int(inputs['total_hashtag_count'])
total_url_count = int(inputs['total_url_count'])
total_mentions_count = int(inputs['total_mentions_count'])
total_media_count = int(inputs['total_media_count'])
number_of_tweets_scraped = int(inputs['number_of_tweets_scraped'])

# (3-3) 판별하기
msg = "none"

if inputs != "":
    data_list = [
        total_retweet_count, total_favorite_count, total_hashtag_count, 
        total_url_count, total_mentions_count, total_media_count, 
        number_of_tweets_scraped
    ]
    
    data_df = pd.DataFrame([data_list])
    
    data_ei = ei_scaler.transform(data_df)
    data_ns = ns_scaler.transform(data_df)
    data_tf = tf_scaler.transform(data_df)
    data_pj = pj_scaler.transform(data_df)
    data_ei = ei_scaler.transform(data_df)
    data_ns = ns_scaler.transform(data_df)
    data_tf = tf_scaler.transform(data_df)
    data_pj = pj_scaler.transform(data_df)
    
    pre_ei = ei_model.predict(data_ei)[0]
    pre_ns = ns_model.predict(data_ns)[0]
    pre_tf = tf_model.predict(data_tf)[0]
    pre_pj = pj_model.predict(data_pj)[0]
    
    msg = f'{pre_ei}{pre_ns}{pre_tf}{pre_pj}'

# (4) 사용자에게 WEB 화면 제공
showHTML(msg, inputs)
