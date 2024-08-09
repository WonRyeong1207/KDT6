# http://www.pythonscraping.com/pages/page1.html

# requests 모듈
import requests     # urllib.request 보다 직관적임(?)
from bs4 import BeautifulSoup

url = 'http://www.pythonscraping.com/pages/page1.html'
html = requests.get(url)

print()
print('html.encoding: ', html.encoding)
print()
print(html.text)

soup = BeautifulSoup(html.text, 'html.parser')
print()
print('h1.string: ', soup.h1.string)

