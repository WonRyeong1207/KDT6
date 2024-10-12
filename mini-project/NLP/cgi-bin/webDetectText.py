# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import torch      # AI 모델 관련
import sys, codecs # 인코딩 관련
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\NLP')
sys.path.append('C:\\Users\\PC\\Desktop\\AI_KDT6\\KDT6\\mini-project\\NLP\\model')
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import work_rnn_func as rnn
from work_rnn_func import BCRNNModels
from work_rnn_func import SentenceClassifier
import pandas as pd
import numpy as np
import re
import json

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 미리 선언하고 가는 것
vocab_df_me = pd.read_csv(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\psychiatry_vocab.csv', encoding='utf-8')
vocab_df_yh = pd.read_csv(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\nose_vocab.csv', encoding='utf-8')

vocab_me = {}
for i in range(len(vocab_df_me)):
    vocab_me[vocab_df_me.loc[i, 'key']] = vocab_df_me.loc[i, 'value']
    
vocab_yh = {}
for i in range(len(vocab_df_yh)):
    vocab_yh[vocab_df_yh.loc[i, 'key']] = vocab_df_yh.loc[i, 'value']

with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\vocab_kdy.json', 'r') as f:
    vocab_team1 = json.load(f)
    
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\vocab_keh.json', 'r') as f:
    vocab_team2 = json.load(f)

vocab_dy = {}
for idx, word in enumerate(vocab_team1):
    vocab_dy[word] = idx

vocab_eh = {}
for idx, word in enumerate(vocab_team2):
    vocab_eh[word] = idx

ko_stopwords = rnn.load_ko_stopwrod()

LABEL_TRANSLATE ={0:'정신과', 1:'이비인후과', 2:'안과', 3:'내과', 4:"그외"}

# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(text, msg):
    print("Content-Type: text/html; charset=utf-8")
    print("Cache-Control: no-cache, no-store, must-revalidate")  # 캐시 방지
    print("Pragma: no-cache")  # HTTP 1.0 캐시 방지
    print("Expires: 0")  # 구형 브라우저 캐시 방지
    # print(f"""
    
    #     <!DOCTYPE html>
    #     <html lang="ko">
    #      <head>
    #       <meta charset="UTF-8">
    #       <title>Text classification</title>
    #      </head>
    #      <body>
    #       <form method="post">
    #         <textarea name="text" rows="10" cols="40">{text}</textarea>
    #         <p><input type="submit" value="classification"></p>
    #       </form>
    #       <p>{msg}의 진료를 받으세요</p>
    #      </body>
    #     </html>""")
    print(f"""
    
        <!DOCTYPE html>
        <html lang="ko">
         <head>
          <meta charset="UTF-8">
          <title>Text Classification</title>
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
            .question-input {{
                width: 100%;
                padding: 15px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 20px;
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
            <h2>질문을 입력하세요:</h2>
            <form method="post">
              <textarea name="text" class="question-input" rows="10" cols="40">{text}</textarea>
              <br>
              <button type="submit" class="submit-btn">질문하기</button>
            </form>
            
            <div class="answer-section">
              <h3>답변:</h3>
              <p>{msg}의 진료를 받으세요</p>
            </div>
          </div>
         </body>
        </html>""")


def transform(text_data, vocab, type_='me'):
    text_data = re.sub('[^ㄱ-ㅎ가-힣]+', ' ', text_data)
    text_data = text_data.replace('\n', '')
    text_data = text_data.replace('\t', '')
    
    if type_ == 'me':
        token_list = []
        morphs = rnn.ko_okt.morphs(text_data)
        for token in morphs:
            if not token in ko_stopwords:
                token_list.append(token)
        
        for token in token_list:
            sent = []
            for token in token_list:
                try:
                    sent.append(vocab[token])
                except:
                    sent.append(vocab['OOV'])
        
        current_len = len(sent)
        if current_len < 50:
            sent.extend([0]*(50-current_len))
        else:
            # sent = sent[current_len-len_data:]
            sent = sent[:50]
            
    elif type_ == 'others':
        token_list = []
        morphs = rnn.ko_okt.morphs(text_data)
        for token in morphs:
            if not token in ko_stopwords:
                token_list.append(token)
        
        for token in token_list:
            sent = []
            for token in token_list:
                try:
                    sent.append(vocab[token])
                except:
                    sent.append(vocab['<unk>'])
        
        current_len = len(sent)
        if current_len < 32:
            sent.extend([0]*(32-current_len))
        else:
            # sent = sent[current_len-len_data:]
            sent = sent[:32]
            
    sent = torch.tensor(sent)
    
    return sent

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    pklfile_me = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\model\bc_lstm_clf_model.pth'
    pklfile_yh = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\model\model_train_all_4_0.9889391654097537.pth'
    pklfile_eh = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\model\model_train_all_2_0.987430869783811.pth'
    pklfile_dy = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\model\KDY_NLP_model_train_wbs.pth'
    
n_vocab_me = len(vocab_me)  # 어휘 사전의 크기s
n_vocab_yh = len(vocab_yh)
n_vocab_dy = len(vocab_dy)
n_vocab_eh = len(vocab_eh)

hidden_dim = 64  # 은닉층의 차원
embedding_dim = 128  # 임베딩 차원
n_layers = 2  # RNN의 레이어 수

lstm_model_me = BCRNNModels(n_vocab=n_vocab_me, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model_me = torch.load(pklfile_me, weights_only=False)
lstm_model_yh = SentenceClassifier(n_vocab=n_vocab_yh, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model_yh = torch.load(pklfile_yh, weights_only=False)
lstm_model_dy = SentenceClassifier(n_vocab=n_vocab_dy, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model_dy = torch.load(pklfile_dy, weights_only=False)
lstm_model_eh = SentenceClassifier(n_vocab=n_vocab_eh, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model_eh = torch.load(pklfile_eh, weights_only=False)


# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태크 속 데이터 가져오기
text = form.getvalue("text", default="")
#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기
msg ="병원"
if text != "":
    text_me = transform(text, vocab_me)
    text_dy = transform(text, vocab_dy, type_='others')
    text_yh = transform(text, vocab_yh, type_='others')
    text_eh = transform(text, vocab_eh, type_='others')

    pred_me = rnn.predict_web(lstm_model_me, text_me, type_='me')
    pred_yh = rnn.predict_web(lstm_model_yh, text_yh)
    pred_dy = rnn.predict_web(lstm_model_dy, text_dy)
    pred_eh = rnn.predict_web(lstm_model_eh, text_eh)
    
    pred_list = [pred_me, pred_yh, pred_dy, pred_eh]
    pred = np.argmax(pred_list)
    pred = LABEL_TRANSLATE[pred]
    
    # result_text= result
    msg = f"{pred}"

# (4) 사용자에게 WEB 화면 제공
showHTML(text,msg)
