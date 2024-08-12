# CSS 소성을 이용한 검색

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
soup = BeautifulSoup(html, 'html.parser')
print()

# 등장인뭉의 이름: 녹색
name_list = soup.find_all('span', {'class' : 'green'})
for name in name_list:
    print(name.string)