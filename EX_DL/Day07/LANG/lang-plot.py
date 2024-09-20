# 모듈로딩 --------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import json
import os.path

# 데이터 변수 선언 --------------------------------------------
DEBUG = True
LANG_CSV= '../DATA/LANG/'

# 언어데이터 파일 생성 ----------------------------------------
if not os.path.exists(LANG_CSV): os.mkdir(LANG_CSV)

# (1) 알파벳 출현 빈도 데이터 수집
with open("../DATA/LANG/freq.json", mode="r", encoding="utf-8") as fp:
    freq = json.load(fp)            # JSON 문자열 데이터 읽어서 리스트 또는 딕셔너리 결과 반환
    if DEBUG: print('freq ===={}\n{}'.format(type(freq), len(freq)))

# (2) 데이터 분석 => 언어마다 계산
lang_dic = {}
if DEBUG:
    print('freq[0]["labels"] = {}'.format(freq[0]["labels"]))
    print('freq[1]["labels"]= {}'.format(freq[1]["labels"]))

for i, lbl in enumerate(freq[0]["labels"]):
    if DEBUG: print('i = {}\t lbl = {}'.format(i, lbl))

    fq = freq[0]["freqs"][i]    # 트레인 용 데이터 추출
    if not (lbl in lang_dic):   # 레이블 언어 데이터 채우기
        lang_dic[lbl] = fq
        continue
    for idx, v in enumerate(fq):                            # 언어별 알파벳 26개 빈도 값 채우기
        lang_dic[lbl][idx] = (lang_dic[lbl][idx] + v) / 2   # 0~1사이 값으로 범위 한정

# (3) Plot을 그리기 위해서 DataFrame에 데이터 넣기
asclist = [[chr(n) for n in range(97,97+26)]]   # 'a' => code value 97
df = pd.DataFrame(lang_dic, index=asclist)

# (4) 그래프 그리기
# ggplot -> grammar of graphics plot
plt.style.use('ggplot')
df.plot(kind="bar", subplots=True, ylim=(0,0.15))
df.plot(kind='line', ylim=(0,0.15))
plt.savefig("lang-plot.png")
plt.show()


