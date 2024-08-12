from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


# 정규 표현식과 BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
soup = BeautifulSoup(html, 'html.parser')
print()

# 대소문자 구분없이 특정 단어 검색
# '[T|t]{t}he prince' : T 또는 t가 1회

princeList = soup.find_all(string='the prince')
print('the prince count: ', len(princeList))

prince_list = soup.find_all(string=re.compile('[T|t]{1}he prince'))
print('T|the prince count: ', len(prince_list))

