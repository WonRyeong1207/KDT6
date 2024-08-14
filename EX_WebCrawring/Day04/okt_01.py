from konlpy.tag import Okt
okt = Okt() # Opem Korean Text (과거 트위터 형태소 분석기)
text = '마음에 꽂힌 칼자루 보다 마음에 꽂힌 꽃한송이가 더 아파서 잠이 오지 않는다'

# pos(text): 문장의 각 품사를 태깅
# norm=True: 문장을 정규화, stem=True: 어간을 추출
okt_tag = okt.pos(text, norm=True, stem=True)
print()
print(okt_tag)
print()

# nouns(text):  명사만 리턴
okt_nouns = okt.nouns(text)
print(okt_nouns)
print()
