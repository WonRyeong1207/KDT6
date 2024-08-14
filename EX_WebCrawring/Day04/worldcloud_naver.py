# 네이버 뉴스 타이틀 wordcloud

from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import platform
import numpy as np
from PIL import Image

# 그 BeutifulSoup 불러오기 오류나서...
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
print()

def get_title(strat_num, end_num, search_word, title_list):
    # start_num ~ end_num까지 크롤링
    while strat_num <= end_num:
        url = ('https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}&start={}'.format(search_word, strat_num))
        
        req = requests.get(url)
        time.sleep(1)
        
        if req.ok:  # 정상적인 request 확인
            soup = BeautifulSoup(req.text, 'html.parser')
            news_titles = soup.find_all('a', {'class':'news_tit'})
            for news in news_titles:
                title_list.append(news['title'])
        
        strat_num += 10
        print('title 개수: ', len(title_list))
        print(title_list)
        print()
        

def make_worldcloud(title_list, stopwords, word_count):
    okt = Okt()
    sentences_tag = []
    
    # 형태소 분석하여 리스트에 넣기
    for sentence in title_list:
        morph = okt.pos(sentence)
        sentences_tag.append(morph)
        print(morph)
        print('-'*80)
    print()
    
    noun_adj_list = []
    # 명사와 형용사, 영단어(Alpha)를 리스트에 추가
    for sentencel in sentences_tag:
        for word, tag in sentencel:
            if tag in ['Noun', 'Adjective', 'Alpha']:
                noun_adj_list.append(word)
                
    # 형태소별 count
    counts = Counter(noun_adj_list)
    tags = counts.most_common(word_count)
    print('-'*80)
    print(tags)
    print()
    
    tag_dict = dict(tags)
    # 검색어 제외 방법 2: dict에서 해당 검색어 제거
    for stopwatd in stopwords:
        if stopwatd in tag_dict:
            tag_dict.pop(stopwatd)
    print(tag_dict)
    print()
    
    if platform.system() == 'Windows':
        path = r'C:\Users\PC\Desktop\AI_빅데이터 전문가 양성과정 6기\KDT6\font\malgun.ttf'
        
    img_mask = np.array(Image.open('./data/cloud.png'))
    wc = WordCloud(font_path=path, width=800, height=600, background_color='white',
                   max_font_size=200, repeat=True, colormap='inferno', mask=img_mask)
    cloud = wc.generate_from_frequencies(tag_dict)
    
    cloud.to_file('./data/news_naver_cloud.jpg')
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()
    
if __name__ == '__main__':
    search_word = 'ChatGTP'
    title_list = []
    stopwards = [search_word, '데이터']
    
    # 1 ~ 200번 게시글까지 크롤링
    get_title(1, 200, search_word, title_list)
    # 단어 50개까지 wordcloud
    make_worldcloud(title_list, stopwards, 50)