from konlpy.tag import Okt

text = """나랏말이 중앙과 달라 한자와 서로 통하지 아니하므로,
        어린 백성들이 말하고 싶은 것이 있어도 마침내 제 뜻을  잘 표현하지 못하는 사람이 많다.
        내 이를 딱하게 여겨 새로 스물여덟 자를 만들었으니,
        사람들로 하여금 쉽게 익히어 날마다 쓰는 데 편하게 할 뿐이다."""
        
okt = Okt()
print()

# morphs(text): 텍스트를 형태소 단위로 나눔
okt_morphs = okt.morphs(text)
print('morphs():', okt_morphs, sep='\n')
print()

# 명사만 추출
okt_nouns = okt.nouns(text)
print('nouns():', okt_nouns, sep='\n')
print()

# phrases(text): 어절 추출
okt_phrases = okt.phrases(text)
print('phrases():', okt_phrases, sep='\n')
print()

# pos(text): 품사를 태깅
okt_pos = okt.pos(text)
print('pos():', okt_pos, sep='\n')
print()

