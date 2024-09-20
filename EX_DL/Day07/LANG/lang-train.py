# 모듈로딩 -------------------------------------------------
import os.path

import joblib as joblib
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 변수 선언 -------------------------------------------
DEBUG = False
FILENAME= '../DATA/LANG/freq.json'
MODLE_NAME='lang.pkl'

# (1) 데이터 준비 --------------------------------------------
data=pd.read_json(FILENAME)
if DEBUG:
    print(f'type  : { type(data) }')
    print(f'Data----\n{ data.head()}')
    data.info()

# (2) 데이터 나누기 ------------------------------------------
train_data=data.loc[0,'freqs']
train_label=data.loc[0,'labels']
test_data=data.loc[1,'freqs']
test_label=data.loc[1,'labels']

print(f'train_data----\n{ type(train_data)}')
print(f'train_data----\n{ len(train_data)}')
print(train_data)
# (3) 학습 및 예측 -----------------------------------------------
clf = svm.SVC()
clf.fit(train_data, train_label)
# predict = clf.predict(test_data)
#
# # (4) 결과 테스트 ------------------------------------------------
# ac_score  = metrics.accuracy_score(test_label, predict)
# cl_report = metrics.classification_report(test_label, predict)
# print("정답률 =", ac_score)
# print("리포트 =")
# print(cl_report)
#
# # (5) 모델 저장 -------------------------------------------------
# if ac_score >=0.99:
#     joblib.dump(clf,MODLE_NAME)
#
# if os.path.exists(MODLE_NAME): print(f'모델 {MODLE_NAME} 저장')