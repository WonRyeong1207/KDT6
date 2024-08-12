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
    for link in bs.find_all('a', href=re.compile('^(/wiki)')):
        if 'href' in link.attrs:
            # 새로운 페이지 발견
            newPage = link.attrs['href']
            count += 1
            print(f"[{count}]: {newPage}")
            pages.add(newPage)
            get_links(newPage)

get_links('')
