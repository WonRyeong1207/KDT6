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
import pandas as pd
import re

# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능

# 미리 선언하고 가는 것
vocab_df = pd.read_csv(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\psychiatry_vocab.csv', encoding='utf8')
vocab = {}
for i in range(len(vocab_df)):
    vocab[vocab_df.loc[i, 'key']] = vocab_df.loc[i, 'value']
ko_stopwords = rnn.load_ko_stopwrod()


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
    print(f"""
    
        <!DOCTYPE html>
        <html lang="ko">
         <head>
          <meta charset="UTF-8">
          <title>Text classification</title>
         </head>
         <body>
          <form>
            <textarea name="text" rows="10" colos="40" >{text}</textarea>
            <p><input type="submit" value="predict">{msg}</p>
          </form>
         </body>
        </html>""")


def transform(text_data):
    text_data = re.sub('[^ㄱ-ㅎ가-힣]+', ' ', text_data)
    text_data = text_data.replace('\n', '')
    text_data = text_data.replace('\t', '')
    
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
    
    for idx, en in enumerate(sent):
        current_len = len(sent)
        if current_len < 50:
            sent.extend([0]*(50-current_len))
        else:
            # sent = sent[current_len-len_data:]
            sent = sent[:50]
            
    sent = torch.tensor(sent)
    
    return sent

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# (2) 모델 로딩
if SCRIPT_MODE:
    pklfile = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\NLP\model\bc_lstm_clf_model.pth'
    
n_vocab = len(vocab)  # 어휘 사전의 크기
hidden_dim = 128  # 은닉층의 차원
embedding_dim = 100  # 임베딩 차원
n_layers = 2  # RNN의 레이어 수

lstm_model = BCRNNModels(n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model = torch.load(pklfile, weights_only=False)

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()

# (3-2) Form안에 textarea 태크 속 데이터 가져오기
text = form.getvalue("text", default="")
#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기
msg =""
if text != "":
    text = transform(text)
    result_text= rnn.predict_web(lstm_model, text)
    msg = f"{result_text}"

# (4) 사용자에게 WEB 화면 제공
showHTML(text,msg)
