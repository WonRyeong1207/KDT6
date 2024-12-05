# Flask Framework에서 '/' URL에 대한 라우팅 처리
# - 파일명: main_views.py

from flask import Blueprint, render_template, request

# 모델 관련
import sys
# model을 못들고 와서 경로추가
sys.path.append(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal')
sys.path.append(r"C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\model")
from seq2_func import Encoder
from seq2_func import Decoder
from seq2_func import Seq2Seq
import seq2_func as seq

import pickle
import re
import torch
import torch.nn as nn

# with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\okt\fs_vocab_okt.pkl', mode='rb') as f:
#     fs_vocab = pickle.load(f)
# with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\okt\ss_vocab_okt.pkl', mode='rb') as f:
#     ss_vocab = pickle.load(f)
# with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\okt\index_fs_vocab.pkl', mode='rb') as f:
#     re_fs_vocab = pickle.load(f)
# with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\okt\index_ss_vocab.pkl', mode='rb') as f:
#     re_ss_vocab = pickle.load(f)
    
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\test\fs_vocab_none.pkl', mode='rb') as f:
    fs_vocab = pickle.load(f)
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\test\ss_vocab_none.pkl', mode='rb') as f:
    ss_vocab = pickle.load(f)
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\test\index_fs_vocab.pkl', mode='rb') as f:
    re_fs_vocab = pickle.load(f)
with open(r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\dict\test\index_ss_vocab.pkl', mode='rb') as f:
    re_ss_vocab = pickle.load(f)
    
# model_file = r'C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\model\third_seq2_model.pth'
# model_file = r"C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\model\third_seq2_param.pth"
model_file = r"C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\model\seq2_test_param.pth"
# model_file = r"C:\Users\PC\Desktop\AI_KDT6\KDT6\mini-project\personal\model\seq2_model.pth"

# hidden_dim, embedding_dim = 128, 64
hidden_dim, embedding_dim = 64, 64
encoder = Encoder(len(fs_vocab), embedding_dim, hidden_dim)
decoder = Decoder(len(ss_vocab), embedding_dim, hidden_dim)
seq2seq_model = Seq2Seq(encoder, decoder)
# seq2seq_model = seq2seq_model.load_state_dict(torch.load(model_file))

state_dict = torch.load(model_file)
# 상태 딕셔너리 로드
if isinstance(state_dict, dict):
    seq2seq_model.load_state_dict(state_dict, strict=False)
else:
    print("state_dict is not a dictionary.")



# Blueprint instance
# http://127.0.0.1:5000/
model_bp = Blueprint('model', import_name='__name__', url_prefix='/', template_folder='templates')

# Routing Functions
# URL 처리
@model_bp.route('/model/', methods=['GET', 'POST'], endpoint='horror')   
# endpoint: rul끝단. 플라스크에서의 의미 url의 끝단이 아닌 그걸 처리하는 함수의 별칭.
# 함수명을 외부에 노출 시키지 않을 수 있음. 내부적으로 함수명을 바꿀 수 있음.
def model_run():
    # 기본값
    text = ''
    msg = '다음 문장을 생성합니다.'
    
    if request.method == 'POST':
        text = request.form.get("text", " ")
        if text:
            enco_in = seq.text_to_sequences([text], fs_vocab)
            print(enco_in)
            enco_pad = seq.padding(enco_in)
            print(enco_pad)
            enco_in_ts = torch.tensor(enco_pad, dtype=torch.long)
            print(enco_in_ts[0])
            input_seq = enco_in_ts[0]

            translated_text = seq.decode_sequence_web(seq2seq_model, input_seq, len(fs_vocab), len(ss_vocab), 50, re_fs_vocab, re_ss_vocab)
            
            msg = f"{translated_text}"
    
    return render_template('model.html', text=text, msg=msg)