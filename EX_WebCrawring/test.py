from urllib.request import urlopen

# html = urlopen('https://www.daangn.com/hot_articles')
# print(type(html))
# print(html.read())


# 패키지 존재여부 확인
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError
import requests


# 그 BeutifulSoup 불러오기 오류나서...
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
    
    
# naver openAPI
# Client ID: y1uqitPVBCd_RQ91I3MJ
# Client Secret: qLwNQvHK1I