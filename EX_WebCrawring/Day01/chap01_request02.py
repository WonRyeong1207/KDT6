# 멜론 웹사이트 접근

from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup

melon_url = 'https://www.melon.com/chart/index.htm'
# HTTP requset 패킷 생성 : Request()
urlrequest = Request(melon_url, headers={'User-Agent':'Mozilla/5.0'})


html = urlopen(urlrequest)
soup = BeautifulSoup(html.read().decode('utf-8'), 'html.parser')    # decode('utf-8'): 한글이 꺠지는 경우 입력해주면 안 깨질 수 있음.
print()
print(soup)
