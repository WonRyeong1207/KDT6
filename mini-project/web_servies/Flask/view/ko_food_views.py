# Flask Framework에서 '/' URL에 대한 라우팅 처리
# - 파일명: main_views.py

from flask import Blueprint, render_template, request
from Flask.models.models import korean_food
from werkzeug.utils import redirect
from Flask import DB

# 모델 예측 관련
import work_rnn_func as rnn
from work_rnn_func import BCRNNModels
import pickle
import re
import torch

# 미리 선언하고 가는 것
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\web_servies\koran_vocab.pkl', 'rb') as f:
    vocab_me = pickle.load(f)

ko_stopwords = rnn.load_ko_stopwrod()

LABEL_TRANSLATE ={0:'초급', 1:"중급"}

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

pklfile_me = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\web_servies\model\bc_lstm_clf_model.pth'

n_vocab_me = len(vocab_me)  # 어휘 사전의 크기s

hidden_dim = 64  # 은닉층의 차원
embedding_dim = 128  # 임베딩 차원
n_layers = 3  # RNN의 레이어 수

lstm_model_me = BCRNNModels(n_vocab=n_vocab_me, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers)   
lstm_model_me = torch.load(pklfile_me, weights_only=False)

# Blueprint instance
# http://127.0.0.1:5000/
ko_food_bp = Blueprint('ko_food', import_name='__name__', url_prefix='/', template_folder='templates')

# Routing Functions
# URL 처리
@ko_food_bp.route('/food_ko/', methods=['GET', 'POST'], endpoint='ko')   
# endpoint: rul끝단. 플라스크에서의 의미 url의 끝단이 아닌 그걸 처리하는 함수의 별칭.
# 함수명을 외부에 노출 시키지 않을 수 있음. 내부적으로 함수명을 바꿀 수 있음.
def ko_food():
    food_list = korean_food.query.all()
    # 판별하기
    text=""
    msg ="난이도"
    
    if request.method == 'POST':
        text = request.form.get("text", "")
        if text:
            # 텐서 변환을 거쳐 문자열로 변환
            text_me = transform(text, vocab_me)
            pred_me = rnn.predict_web(lstm_model_me, text_me, type_='me')
            pred_me = 1 if pred_me > 0.5 else 0
            
            # 문자열로 변환
            text_me_str = str(text)
            
            # 'feature' 칼럼에서 text_me_str를 포함하는 레코드 가져오기
            # food_list = korean_food.query.filter(korean_food.feature.contains(text_me_str)).all()
            
            # 'feature' 칼럼에서 text_me_str를 포함하고, level이 pred_me와 같은 레코드 가져오기
            food_list = korean_food.query.filter(
                korean_food.feature.contains(text_me_str),
                korean_food.level_code == pred_me  # 레벨 체크
            ).all()

            # 결과 출력
            pred = LABEL_TRANSLATE[pred_me]
            msg = f"{pred}"
            print(food_list)
        
    return render_template('ko_food.html', text=text, msg=msg, food_list=food_list)

