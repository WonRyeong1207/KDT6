# 특정 단어 찾기

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
soup = BeautifulSoup(html, 'html.parser')
print()

prince_list = soup.find_all(string='the prince')
print(prince_list)
print('the prince count: ', len(prince_list))
