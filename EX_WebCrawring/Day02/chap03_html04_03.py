# 트리 이동: 형제 다루기

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
soup = BeautifulSoup(html, 'html.parser')
print()

# next_siblings 속성
# 여기는 교수님 결과와 다르게 나옴
for sibling in soup.find('table', {'id':'giftList'}).tr.next_siblings:
    print(sibling)
    print('-'*30)
    
# previous_siblings 속성
print()
print('previous_siblings')
for sibling in soup.find('tr', {'id':'gift2'}).previous_siblings:
    print(sibling)
    
# next_sibling, previous_sibling
# 여기도...
print()
sibling1 = soup.find('tr', {'id':'gift3'}).next_siblings
print('sibling1: ', sibling1)
# print(ord(sibling1))    # ord(문자): 문자의 Unicode 정수를 리턴

sibling2 = soup.find('tr', {'id':'gift3'}).next_sibling.next_siblings
print(sibling2)
print()

# parent 사용
style_tag = soup.style
print(style_tag.parent)
print()

img1 = soup.find('img', {'src':'../img/gifts/img1.jpg'})
text = img1.parent.previous_sibling.get_text()
print(text)