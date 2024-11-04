"""
Seq2Seq model functions & class
 - datasets: translate_ko.txt
 - data: Reddit two sentense horror
 - frame work: Pytorch
"""

import re
import os
import unicodedata
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optima
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchinfo import summary

import string
import spacy
import soynlp
import konlpy
from konlpy.tag import Okt

# 미리 선언 하는 것들
DATA_PATH = './data/translate_ko.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 28
NUM_DATA = 3300
torch.manual_seed(RANDOM_STATE)

ko_model = 'ko_core_news_sm'
ko_spacy = spacy.load(ko_model)
ko_okt = Okt()
ko_soynlp = soynlp.word.WordExtractor()
punct = string.punctuation


# 전처리 함수
def sent_prepro(sent):
    # 공백을 먼저 주면... 
    # 구두점.. 공백을 넣어야하는데..
    sent = re.sub(r'([.!?,"])', r' \1', sent)
    sent = re.sub(r"([':])", r" \1", sent)
    
    # 한국어만 남길 수는 없으니...
    # 음.. 어카지?
    sent = re.sub(r"'", r'"', sent) # 다음 모델은 이렇게 전처리 해보자
    sent = re.sub(r'[^0-9ㄱ-ㅎ가-힣!.?,"]+', r" ", sent)
    
    # 공백 문제를 해결해보려는 시도
    sent = re.sub(r"\s+", " ", sent)
    
    return sent


# 형태소 분석기
def nlp_tokenize(data, nlp_type='None'):
    # 토큰을 담을 그릇
    word_list = []
    
    if nlp_type == 'None':
        pass
    
    elif nlp_type == ko_spacy:
        for sent in data:
            doc = ko_spacy(sent)
            token_list = []
            for token in doc:
                token_list.extend(token)
        # yield token_list
    
    elif nlp_type == ko_okt:
        for sent in data:
            morphs = ko_okt.morphs(sent)
            # carry = []
            # for word in morphs:
            #     carry.append(morphs)
            # word_list.append(carry)
            word_list.extend(morphs)
        # yield word_list
            
    elif nlp_type == ko_soynlp:
        for sent in data:
            ko_soynlp.train(sent)
            for sent in sent:
                result = ko_soynlp.extract()
                score = {w:s.cohesion_forward for w, s in result.items()}
                tokenizer = soynlp.tokenizer.LTokenizer(score)
                carry = []
                for word in tokenizer.tokenize(sent):
                    carry.extend(word)
                word_list.extend(carry)
        
    yield word_list
    
    
# txt에서 데이터를 불러옴
def data_load(data, i):
    data_ = []
    for k in range(4):
        data_.append(data.readline())
    idx = data_[0]
    first = data_[1]
    second = data_[2]
    skip = data_[3]
    yield idx, first, second, skip
    

# 파일 데이터를 가져오는 함수
def load_data(data_path, nlp_type=None):
    #  dict 형태면 좀 편할까?
    data_dict = {'index':[], 'first_sent':[], 'second_sent':[],
                 'encoder_in':[], 'decoder_in':[], 'decoder_target':[]}
    
    with open(data_path, mode='r', encoding='utf-8') as f:
        form = """
        [index] - 1
        XXXX
        XXXX
        
        """
        
        for i in range(NUM_DATA+1):
        
            # form의 형태처럼 생겼으니 음...
            for carry in data_load(data=f, i=i):
                idx, first, second, skip = carry[0], carry[1], carry[2], carry[3]
                
                # print(f"index: {idx}")
                # print(f"first sent: {first}")
                # print(f"second sent: {second}")
                # print(f"skip: {skip}")
                
                # 문장 전처리
                first = sent_prepro(first)
                second = sent_prepro(second)
                
                # print(f"first sent: {first}")
                # print(f"second sent: {second}")

                # NLP 토큰화 적용 (한 문장이 하나의 리스트로 유지되도록 설정)
                if nlp_type == ko_spacy:
                    first = [token for token in nlp_tokenize([first], nlp_type=ko_spacy)]
                    second = [token for token in nlp_tokenize([second], nlp_type=ko_spacy)]
                    
                elif nlp_type == ko_okt:
                    first = [ko_okt.morphs(first)]
                    second = [ko_okt.morphs(second)]
                
                elif nlp_type == ko_soynlp:
                    first = [token for token in nlp_tokenize([first], nlp_type=ko_soynlp)]
                    second = [token for token in nlp_tokenize([second], nlp_type=ko_soynlp)]
                
                # 토큰 리스트에 시작/끝 토큰 추가
                if nlp_type is None:
                    # None 타입의 경우 문자열로 유지
                    second_in = ["<sos>"] + second.split()
                    second_out = second.split() + ["<eos>"]
                    first_in = first.split()
                else:
                    # NLP 토큰화된 경우 리스트로 유지
                    second_in = ["<sos>"] + second[0]  # 리스트 내부 리스트로 유지
                    second_out = second[0] + ["<eos>"]
                    first_in = first[0]
            
                # dict에 넣어버리기
                data_dict['index'].append(idx)
                data_dict['first_sent'].append(first)
                data_dict['second_sent'].append(second)
                data_dict['encoder_in'].append(first_in)
                data_dict['decoder_in'].append(second_in)
                data_dict['decoder_target'].append(second_out)
                
        return data_dict
    

# 단어 사전 저장
def bulid_vocab(data, file_name='vocab.pkl', reverse_index=True):
    # 토큰을 담을 그릇
    word_list = []
    
    for sent in data:
        for word in sent:
            word_list.append(word)
   
    # print(word_list, '\n')         
    word_counts = Counter(word_list)
    sorted_word = sorted(word_counts, key=word_counts.get, reverse=True)
    
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<OOV>'] = 1
    
    for idx, word in enumerate(sorted_word):
        vocab[word] = idx+2
    
    with open(file_name, mode='wb') as f:
        pickle.dump(vocab, f)
    
    if reverse_index:
        re_name = 'reverse_'+file_name
        re_vocab = {v:k for k, v in vocab.items()}
        with open(re_name, mode='wb') as f:
            pickle.dump(re_vocab, f)
        return vocab, re_vocab
    
    else:
        return vocab
    
    
# encoding
def text_to_sequences(sents, vocab):
    encoded_X_data = []
    for sent in tqdm(sents):
        idx_seq = []
        for word in sent:
            try:
                idx_seq.append(vocab[word])
            except:
                idx_seq.append(vocab['<OOV>'])
        encoded_X_data.append(idx_seq)
    return encoded_X_data


# 이제 패딩...
def padding(sents, max_length=None):
    # 최대 길이 값이 주어지지 않을 경우 데이터 내 최대 길이로 패딩
    if max_length is None:
        max_length = max([len(sent) for sent in sents])
        
    features = np.zeros((len(sents), max_length), dtype=int)
    for idx, sent in enumerate(sents):
        if len(sent) != 0:
            features[idx, :len(sent)] = np.array(sent)[:max_length]
    return features


# encoder class
class Encoder(nn.Module):
    def __init__(self, fs_vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(fs_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        
    def forward(self, x):
        y = self.embedding(x)
        _, (hidden_state, cell_state) = self.lstm(y)
        # 인코더의 출력은 hidden_state, cell_state
        return hidden_state, cell_state
    

# decoder class
class Decoder(nn.Module):
    def __init__(self, ss_vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(ss_vocab_size, embedding_dim, padding_idx=0)
        # 여기 lstm 파트를 custom 하면 성능이 좋아질까?
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        # 여기는 출력층
        self.fc = nn.Linear(hidden_units, ss_vocab_size)
        
    def forward(self, x, hidden_state, cell_state):
        # 인코더의 입력을 받아야 하니까
        y = self.embedding(x)
        
        # 이제 인코더의 결과값을 받음
        output, (hidden, cell) = self.lstm(y, (hidden_state, cell_state))
        output = self.fc(output)
        
        return output, hidden, cell
    

# model class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # model에서는 인코더와 디코더의 결과를 이제 합치는 겨?
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, inputs, target):
        hidden, cell = self.encoder(inputs)
        
        # 3D로 필요한 경우에만 unsqueeze(0) 추가
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)

        # 디코더에 hidden과 cell을 전달
        output, hidden_state, cell_state = self.decoder(target, hidden, cell)
        
        return output
    

# 손실 계산하는 함수?
# loss와 acurassy 계산하는 함수
def calu(model, data_loader):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for encoder_inputs, decoder_inputs, decoder_targets in data_loader:
        encoder_inputs = encoder_inputs.to(DEVICE)
        decoder_inputs = decoder_inputs.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        outputs = model(encoder_inputs, decoder_inputs)

        # 손실 값 계산
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
        total_loss += loss.item()  # 손실 값을 숫자 값으로 사용

        # 정확도 계산 (패딩 토큰 제외)
        mask_data = (decoder_targets != 0)
        predictions = outputs.argmax(dim=-1)
        correct_predictions = (predictions == decoder_targets) & mask_data
        total_correct += correct_predictions.sum().item()
        total_samples += mask_data.sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def training(model, train_datasets, test_datasets, epochs=100, BATCH_SIZE=64, lr=0.001, patience=10):
    
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    
    model = model.to(DEVICE)
    optimizer = optima.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    train_dl = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(test_datasets, batch_size=10
                        , shuffle=True)
    
    save_model = './model/seq2_model.pth'
    save_param = './model/seq2_param.pth'
    
    for epoch in range(1, epochs+1):
        model.train()
        
        for encoder_inputs, decoder_inputs, decoder_targets in train_dl:
            encoder_inputs = encoder_inputs.to(DEVICE)
            decoder_inputs = decoder_inputs.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            optimizer.zero_grad()
            
            # 순전파 전파
            outputs = model(encoder_inputs, decoder_inputs)
            
            # loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            loss.backward()
            
            optimizer.step()
            
        train_loss, train_acc = calu(model, train_dl)
        val_loss, val_acc = calu(model, val_dl)
        
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
    
        if epoch%5 == 0:
                print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc*100:.6f} %")
                print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_acc*100:.6f} %\n")
            
        if len(acc_dict['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_dict['val'][-1] >= max(acc_dict['val']):
                print(f"[{epoch:5}/{epochs:5}] saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                    
        scheduler.step(val_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc*100:.6f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_acc*100:.6f} %\n")
            break

    return loss_dict, acc_dict


def over_training(model, train_datasets, test_datasets, epochs=100, BATCH_SIZE=64, lr=0.001, patience=10):
    
    loss_dict = {'train':[], 'val':[]}
    acc_dict = {'train':[], 'val':[]}
    
    model = model.to(DEVICE)
    optimizer = optima.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')
    train_dl = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(test_datasets, batch_size=10, shuffle=True)
    
    save_model = './model/seq2_over_model.pth'
    save_param = './model/seq2_over_param.pth'
    
    for epoch in range(1, epochs+1):
        model.train()
        
        for encoder_inputs, decoder_inputs, decoder_targets in train_dl:
            encoder_inputs = encoder_inputs.to(DEVICE)
            decoder_inputs = decoder_inputs.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            optimizer.zero_grad()
            
            # 순전파 전파
            outputs = model(encoder_inputs, decoder_inputs)
            
            # loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            loss.backward()
            
            optimizer.step()
            
        train_loss, train_acc = calu(model, train_dl)
        val_loss, val_acc = calu(model, val_dl)
        
        loss_dict['train'].append(train_loss)
        loss_dict['val'].append(val_loss)
        acc_dict['train'].append(train_acc)
        acc_dict['val'].append(val_acc)
    
        if epoch%5 == 0:
                print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc*100:.6f} %")
                print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_acc*100:.6f} %\n")
            
        if len(acc_dict['val']) == 1:
            print("saved first")
            torch.save(model.state_dict(), save_param)
            torch.save(model, save_model)
        else:
            if acc_dict['val'][-1] >= max(acc_dict['val']):
                print(f"[{epoch:5}/{epochs:5}] saved model")
                torch.save(model.state_dict(), save_param)
                torch.save(model, save_model)
                    
        scheduler.step(train_loss)
        
        if scheduler.num_bad_epochs >= scheduler.patience:
            print('성능 및 손실의 개선이 없어서 학습을 중단합니다.\n')
            print(f"[{epoch:5}/{epochs:5}]  [Train]         loss: {train_loss:.6f}, score: {train_acc*100:.6f} %")
            print(f"[{epoch:5}/{epochs:5}]  [Validation]    loss: {val_loss:.6f}, score: {val_acc*100:.6f} %\n")
            break

    return loss_dict, acc_dict


# train 결과 보여주기
def draw_two_plot(loss, score, title, type_='LSTM'):
    
    # 축을 2개 사용하고 싶음.
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()
    
    ax1.plot(loss['train'], label=f"train loss mean: {sum(loss['train'])/len(loss['train']):.6f}", color='#5587ED')
    ax1.plot(loss['val'], label=f"validation loss mean: {sum(loss['val'])/len(loss['val']):.6f}", color='#F361A6')
    ax2.plot(score['train'], label=f"train score max: {max(score['train'])*100:.2f} %", color='#00007F')
    ax2.plot(score['val'], label=f"validation score max: {max(score['val'])*100:.2f} %", color='#99004C')
    
    fig.suptitle(f'{title} {type_} model', fontsize=15)
    ax1.set_ylabel('loss', fontsize=10, color='#5587ED')
    ax2.set_ylabel('score', fontsize=10, color='#00007F')
    
    fig.legend(fontsize='small', loc='lower left')
    # plt.xticks(np.arange(0, len(loss['train']), 2), labels=[x for x in range(1, len(loss['val'])+1, 2)])
    plt.show()
    

# 인코딩을 디코딩!!
def seq_to_enco(input_seq, re_vocab):
    sent = ''
    for enco_word in input_seq:
        if (enco_word != 0):
            sent = sent + re_vocab[enco_word] + ' '
    return sent

def seq_to_deco(input_seq, vocab, re_vocab):
    sent =''
    for enco_word in input_seq:
        if(enco_word != 0 and enco_word != vocab['<sos>'] and enco_word != vocab['<eos>']):
            sent = sent + re_vocab[enco_word] + ' '
    return sent


# predict? 한걸 보여주는 것
def decode_sequence(model, input_seq, src_vocab_size, tar_vocab_size, max_output_len, int_to_src_token, int_to_tar_token):
    encoder_inputs = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 인코더의 초기 상태 설정
    hidden, cell = model.encoder(encoder_inputs)

    # 시작 토큰 <sos>을 디코더의 첫 입력으로 설정
    # unsqueeze(0)는 배치 차원을 추가하기 위함.
    decoder_input = torch.tensor([3], dtype=torch.long).unsqueeze(0).to(DEVICE)

    decoded_tokens = []

    # for문을 도는 것 == 디코더의 각 시점
    for _ in range(max_output_len):
        output, hidden, cell = model.decoder(decoder_input, hidden, cell)

        # 소프트맥스 회귀를 수행. 예측 단어의 인덱스
        output_token = output.argmax(dim=-1).item()

        # 종료 토큰 <eos>
        if output_token == 4:
            break

        # 각 시점의 단어(정수)는 decoded_tokens에 누적하였다가 최종 번역 시퀀스로 리턴
        decoded_tokens.append(output_token)

        # 현재 시점의 예측. 다음 시점의 입력으로 사용된다.
        decoder_input = torch.tensor([output_token], dtype=torch.long).unsqueeze(0).to(DEVICE)

    return ' '.join(int_to_tar_token[token] for token in decoded_tokens)


def decode_sequence_web(model, input_seq, src_vocab_size, tar_vocab_size, max_output_len, int_to_src_token, int_to_tar_token):
    # 이미 텐서라면 unsqueeze(0)만 적용하여 3D로 변경
    if isinstance(input_seq, torch.Tensor):
        encoder_inputs = input_seq.unsqueeze(0).to(DEVICE)
    else:
        encoder_inputs = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 인코더의 초기 상태 설정
    hidden, cell = model.encoder(encoder_inputs)

    # 시작 토큰 <sos>을 디코더의 첫 입력으로 설정
    decoder_input = torch.tensor([3], dtype=torch.long).unsqueeze(0).to(DEVICE)

    decoded_tokens = []

    # for문을 도는 것 == 디코더의 각 시점
    for _ in range(max_output_len):
        output, hidden, cell = model.decoder(decoder_input, hidden, cell)

        # 소프트맥스 회귀를 수행. 예측 단어의 인덱스
        output_token = output.argmax(dim=-1).item()

        # 종료 토큰 <eos>
        if output_token == 4:
            break

        # 각 시점의 단어(정수)는 decoded_tokens에 누적하였다가 최종 번역 시퀀스로 리턴
        decoded_tokens.append(output_token)

        # 현재 시점의 예측. 다음 시점의 입력으로 사용된다.
        decoder_input = torch.tensor([output_token], dtype=torch.long).unsqueeze(0).to(DEVICE)

    return ' '.join(int_to_tar_token[token] for token in decoded_tokens)

