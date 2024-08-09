# 멜론 웹사이트 접근

from urllib.request import urlopen
from bs4 import BeautifulSoup

# 샘플코드 1
# urllib.error.HTTOError: HTTP Error 406: Not Acceptable 발생

melon_url = 'https://www.melon.com/chart/index.htm'
html = urlopen(melon_url)   # 사용자 정보가 없기 떄문에 발생하는 에러

soup = BeautifulSoup(html.read(), 'html.parser')
print()
print(soup)
