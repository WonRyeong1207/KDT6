from urllib.request import urlopen

# html = urlopen('https://www.daangn.com/hot_articles')
# print(type(html))
# print(html.read())


# 패키지 존재여부 확인
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError
import requests