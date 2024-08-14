# 단어 분석 및 Word Cloud 생성

from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
import platform     # 폰트를 지정하기 위함
import koreanize_matplotlib     # 폰트가 깨질수 있음
import numpy as np
from PIL import Image # 이건 대문자로 적어야 인식됨

text = open('./data/test.txt', encoding='utf-8').read()
okt = Okt()     # 객체 생성
print()

# okt함수를 통해 읽어들인 내용의 형태소를 분석
sentences_tag = []
sentences_tag = okt.pos(text)

noun_adj_list = []

# tag가 명사이거나 형용사이거나 단어들만 noun_adj)list에 넣음
for word, tag in sentences_tag:
    if tag in ['Noun', 'Adjective']:
        noun_adj_list.append(word)
        
# 가장 많이 나온 단어부터 50개를 저장
counts = Counter(noun_adj_list)
tags = counts.most_common(50)
print(tag)
print()

# 한글을 분석하기 위해 font를 한글로 지정.
if platform.system() == 'Windows':
    path = r'C:\Users\PC\Desktop\AI_빅데이터 전문가 양성과정 6기\KDT6\font\malgun.ttf'

img_mask = np.array(Image.open('./data/cloud.png'))
wc = WordCloud(font_path=path, width=40, height=400,
               background_color='white', max_font_size=200,
               repeat=True, colormap='inferno', mask=img_mask)
cloud = wc.generate_from_frequencies(dict(tags))

# 생성된 WordColud를 test.jpg로 저장
cloud.to_file('./data/test.jpg')
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(cloud)
plt.show()
