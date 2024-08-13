# 파이썬과 통합: 위키피디아 자료를 mysql 저장

from urllib.request import urlopen
from bs4 import BeautifulSoup
import random
import pymysql
import re

import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# 데이터베이스에 저장
def store(conn, cur, title, content):
    cur.execute('insert into pages (title, content) values("%s", "%s")', (title, content))
    conn.commit()
    
# 데이터 크롤링
def get_links(conn, cur, article_url):
    html = urlopen('https://en.wikipedia.org'+article_url)
    bs = BeautifulSoup(html, 'html.parser')
    
    title = bs.find('h1').text
    content = bs.find('div', {'id':'mw-content-text'}).find('p').text
    print(title, content)
    
    # save
    store(conn, cur, title, content)
    
    return bs.find('div', {'id':'bodyContent'}).find_all('a', href=re.compile('^(/wiki/((?!:).)*$)'))

def main():
    conn = pymysql.connect(host='localhost', user='test',
                           passwd='test1111', db='scraping', charset='utf8')
    cur = conn.cursor()
    random.seed(None)
    
    links = get_links(conn, cur, '/wiki/Kevin_Bacon')
    
    try:
        while len(links) > 0:
            newArticle = links[random.randint(0, len(links)-1)].attrs['href']
            print(newArticle)
            links = get_links(conn, cur, newArticle)
    finally:
        cur.close()
        conn.close()
        
main()