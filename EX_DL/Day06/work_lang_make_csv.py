# 언어 분류 모델
# - make csv file: lang_train.csv, lang_test.csv


# columns: a ~ Z, 소문자만 사용할 예정
# label class: en, fr, id, tl

import os
import pandas as pd
import numpy as np


# 파일 안의 파일 및 폴더를 가져와서 알파벳만 저장하기
def make_df(path):
    files = os.listdir(path)
    label_list = []
    colnames = [chr(x) for x in range(ord('a'), ord('z')+1)]
    data_df = pd.DataFrame(columns=colnames)
    
    for file_ in files:
        label = file_[:2]
        label_list.append(label)
        
        with open(path+file_, mode='r', encoding='utf-8') as f:
            all_data = f.read()
            # 소문자통일
            all_data = all_data.lower()
            all_data = all_data.replace('\n', '')
            
            # 알파벳이 아닌 것들 제거
            for char in all_data:
                if not (ord('a') <= ord(char) <= ord('z')):
                    all_data = all_data.replace(char, '')
                    
            # a ~ z count
            count_data = {key:0 for key in colnames}
            
            for ch in all_data:
                if ch in all_data:
                    count_data[ch] = count_data[ch] + 1
                else:
                    count_data[ch] = 1
                    
            # frequency
            for data in count_data:
                if count_data[data] == 0:
                    count_data[data] = 0
                else:
                    count_data[data] = count_data[data] / len(all_data)
                count_data = dict(sorted(count_data.items()))
                count_data_df = pd.DataFrame(count_data.values(), index=count_data.keys()).T
            
        data_df = pd.concat([data_df, count_data_df], ignore_index=True)
    data_df['language'] = label_list
    
    return data_df

if __name__ == '__main__':
    train_path = '../data/Language/train/'
    test_path = '../data/language/test/'
    
    train_csv = '../data/Language/lang_train.csv'
    test_csv = '../data/Language/lang_test.csv'
    
    train_df = make_df(train_path)
    train_df.to_csv(train_csv, index=False, encoding='utf-8')
    
    test_df = make_df(test_path)
    test_df.to_csv(test_csv, index=False, encoding='utf-8')