# Web Crawring

# 위키피디아 페이지 가져오기
# rul = 'https://en.wikipedia.org/wiki/Kevin_Bacon'

from urllib.request import urlopen
from bs4 import BeautifulSoup
import random
import re

# 링크 무작위 이동
# random.seed(detetime.detetime.now())
random.seed(None)   # python 3.9 이상

def get_link(articleUrl):
    html = urlopen('https://en.wikipedia.org' + articleUrl)
    bs = BeautifulSoup(html, 'html.parser')
    bodyContent = bs.find('div', {'id':'bodyContent'})
    wikiUrl = bodyContent.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
    return wikiUrl

links = get_link('/wiki/Kevin_Bacon')
print('links 길이: ', len(links))

while len(links) > 0:
    newArticle = links[random.randint(0, len(links))-1].attrs['href']
    print(newArticle)
    links = get_link(newArticle)
    
