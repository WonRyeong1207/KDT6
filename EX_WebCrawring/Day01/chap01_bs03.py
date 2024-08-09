# BeautifulSoup 라이브러리

from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

# 존재하지 않는 태그 예외처리
def getTitle(url, tag):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    
    try:
        bsObj = BeautifulSoup(html.read(), 'html.parser')
        value = bsObj.body.find(tag)
    except AttributeError as e:
        return None
    
    return value

tag = 'h2'
value = getTitle('http://www.pythonscraping.com/pages/page1.html', tag)

print() # 출력값을 좀 더 보기 편하게 하기 위해서
if value == None:
    print(f"{tag} could not be found")
else:
    print(value)