# BeautifulSoup 라이브러리

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page1.html')
bs = BeautifulSoup(html.read(), 'html.parser')  # 클래스의 생성자, 객체생성

print()
print(bs)
print()
print(bs.h1)    # 그 꺾새기호까지 들고 오는 것, <h1> text </h1>
print(bs.h1.string) # 기존문법: .text, 지금 쓰라고 하는 string도 문제는 존재
print()
