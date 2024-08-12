# 트리 이동: 자식과 자손

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
soup = BeautifulSoup(html, 'html.parser')
print()

table_tag = soup.find('table', {'id':'giftList'})
print('children 개수: ', len(list(table_tag.children)))

index = 0
for child in table_tag.children:
    index += 1
    print(f"[{index}]: {child}")
    print('-'*30)
    