# KoNLPY 사용법

# pos(텍스트, [norm, stem]): 품사 정보를 부착하여 반환
# - norm: 정규화 여부, stem: 어간 찾기 여부
# morphs(텍스트): 텍스트에서 형태소를 반환
# nouns(텍스트): 텍스트에서 명사만 반환
# phrasees(텍스트): 텍스트에서 어절을 반환

from konlpy.tag import Okt
okt = Okt()

list1 = okt.pos('아버지 가방에 들어가신다.', norm=True, stem=True)
list2 = okt.pos('아버지 가방에 들어가신다.', norm=False, stem=False)
print()
print(list1)
print(list2)
print()

word1 = okt.pos('그래요ㅋㅋ?', norm=True, stem=True)
word2 = okt.pos('그래욬ㅋㅋ?', norm=False, stem=True)
word3 = okt.pos('그래욬ㅋ?', norm=True, stem=False)
print(word1)
print(word2)
print(word3)

