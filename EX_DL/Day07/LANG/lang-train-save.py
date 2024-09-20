# 모듈 로딩 -----------------------------------------
from sklearn import svm
import joblib, json

# 데이터 변수 선언 ------------------------------------------------------
DEBUG = True

# 각 언어의 출현 빈도 데이터(JSON) 읽어 들이기
with open("../../DATA/LANG/freq.json", "r", encoding="utf-8") as fp:
    d = json.load(fp)           # JSON 문자열 데이터 =>  딕셔너리 또는 리스트 타입 변환
    if DEBUG: print('d ===>{}\n{}'.format(type(d), len(d)))
    data = d[0]                 # train용 데이터 가져오기

# 데이터 학습하기
clf = svm.SVC(gamma='scale')
clf.fit(data["freqs"], data["labels"])

# 학습 데이터 저장하기
# joblib
joblib.dump(clf, "../../DATA/LANG/freq.pkl")
print("SAVE OK")

