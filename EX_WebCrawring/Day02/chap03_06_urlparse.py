# 인터넷 크롤링 URL 구조

from urllib.parse import urlparse

url_string1 = 'https://shopping.naver.com/home/p/index.naver'

url = urlparse(url_string1)
print(url.scheme)
print(url.netloc)
print(url.path)

