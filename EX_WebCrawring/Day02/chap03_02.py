# Web Crawring

# 위키피디아 페이지 가져오기
# rul = 'https://en.wikipedia.org/wiki/Kevin_Bacon'

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

html = urlopen('https://en.wikipedia.org/wiki/Kevin_Bacon')
bs = BeautifulSoup(html, 'html.parser')
print()

# 연관 기사 링크 찾기

# 연관 기사의 3가지 특성을 이용:
#  - 정규식: ^(/wiki/)((?!:).)*$
#   - ^: 정규식 시작, $: 정규식 끝
#   - (/wiki/): '/wiki/'문자열 포함
#   - ((?!:).)*$: ':'이 없는 문자열 및 임의의 문자(.)가 0회 이상(*) 반복되는 문자열 검색

body_content = bs.find('div', {'id':'bodyContent'})

# pattern = '^(/wiki/)((?!:).)*$'
# for문이 문법 속성 에러남. 나중에 다시보자
for link in body_content.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$')):
    if 'href' in link.attrs:
        print(link.attrs['href'])
        
# for link in bs.find_all('a'):
#     if 'href' in link.attrs:
#         print(link.attrs['href'])
        
