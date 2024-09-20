import os, joblib

modelFile='./lang.pkl'

if not os.path.exists(modelFile):
    print('Model 파일이 없습니다.')
else:
    langModel=joblib.load(modelFile)
    langModel.predict([])
