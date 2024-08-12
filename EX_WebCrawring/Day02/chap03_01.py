# Web Crawring

# 위키피디아 페이지 가져오기
# rul = 'https://en.wikipedia.org/wiki/Kevin_Bacon'

# 임의의 위키피디아 페이지에서 모든 링크 목록 가져오기
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://en.wikipedia.org/wiki/Kevin_Bacon')
bs = BeautifulSoup(html, 'html.parser')
for link in bs.find_all('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])
        
