# CSS 속성을 이용한 검색

from bs4 import BeautifulSoup

html_text = '<span class="red">Heavens! What a virulent attack!</span>'
soup = BeautifulSoup(html_text, 'html.parser')

object_tag = soup.find('span')
print()
print('object_tag: ', object_tag)
print('attrs: ', object_tag.attrs)
print('value: ', object_tag.attrs['class'])
print('text: ', object_tag.text)