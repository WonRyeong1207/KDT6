# Web Crawring

# 위키피디아 페이지 가져오기
# rul = 'https://en.wikipedia.org/wiki/Kevin_Bacon'

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

# 전체 사이트에서 데이터 수집

pages = set()
count = 0

def get_links(page_url):
    global pages
    global count
    html = urlopen('https://en.wikipedia.org{}'.format(page_url))
    bs = BeautifulSoup(html, 'html.parser')
    
    try:
        print(bs.h1.get_text()) # <h1> 태그 검색
        # print(bs.find(id='hw-content-text').find('p').text)
        print(bs.find('div', attrs={'id':'mw-content-text'}).find('p').text)
    except AttributeError as e:
        print('this page is missing something! Continuing: ', e)
        
    pattern = '^(/wiki/)((?!:).)*$'
    for link in bs.find_all('a', href=re.compile(pattern)):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                newPage = link.attrs['href']
                print('-'*40)
                count += 1
                print(f"[{count}]: {newPage}")
                pages.add(newPage)
                get_links(newPage)

get_links('')
