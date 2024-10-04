import konlpy
from konlpy.tag import Okt

okt = Okt()

sentense = "몸 아파서 디지겠다. 와... ㅈㄴ 아파 ㅠㅜㅠㅜㅠㅜ"

nouns = okt.nouns(sentense)
phrases = okt.phrases(sentense)
morphs = okt.morphs(sentense)
pos = okt.pos(sentense)

print(f"명사 추출: {nouns}")
print(f"구 추출: {phrases}")
print(f"형태소 추출: {morphs}")
print(f"품사 태깅: {pos}")

# 설치가 꼬였는데... 사용에는 지장ㅇ 없네