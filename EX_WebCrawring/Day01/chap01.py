# 당근 마켓의 정보가져오기?

# urllib.request.urlopen(url) 
# : 해당 url에서 파일이나 이미지 파일, 기타파일을 가져오는 함수

from urllib.request import urlopen

html = urlopen('https://www.daangn.com/hot_articles')
print()
print(type(html))
print()
print(html.read())
