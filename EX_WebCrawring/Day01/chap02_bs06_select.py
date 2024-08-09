html_example = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BeautifulSoup 활용</title>
</head>
<body>
    <h1 id="heading">Heading 1</h1>
    <p>Paragraph</p>
    <span class="red">BeautifulSoup Library Examples!</span>
    <div id="link">
        <a class="external_link" href="www.google.com">google</a>

        <div id="class1">
            <p id="first">class1's first paragraph</p>
            <a class="exteranl_link" href="www.naver.com">naver</a>

            <p id="second">class1's second paragraph</p>
            <a class="internal_link" href="/pages/page1.html">Page1</a>
            <p id="third">class1's third paragraph</p>
        </div>
    </div>
    <div id="text_id2">
        Example page
        <p>g</p>
    </div>
    <h1 id="footer">Footer</h1>
</body>
</html>
'''

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_example, 'html.parser')
print()


# select(): 조건에 맞는 모든 태그를 리턴
# select_one(): 조건에 맞는 첫 번째 태그만 리턴
# - 하위 태그를 찾을 때, 직접 하위 경로를 지정

# select_one(): find()와 동일 기능
head = soup.select_one('head')
print(head)
print('head.text: ', head.text.strip())
print()

# 첫 번째 <h1> 태그 검색
h1 = soup.select_one('h1')
print(h1)
print()

# id 검색: #id
# <h1> 태그의 id가 'footer'인 항목 추출
footer = soup.select_one('h1#footer')
print(footer)
print()

# class 검색: .class
# <a class="internal_link" href="/pages/page1.html">Page1</0a>  검색
class_link = soup.select_one('a.internal_link')
print(class_link)
print()
print(class_link.string)
print(class_link['href'])
print()

# 계층적 하위 태그 접근 : 태그가 단계적으로 존재할 때
link1 = soup.select_one('div#link > a.external_link')
print(link1)
print()
# find() 함수와 비교
link_find = soup.find('div', {'id':'link'})
external_link = link_find.find('a', {'class':'external_link'})
print('find external_link: ', external_link)

# 계층적 하위 태그 접근 : 공백으로 하위 태그 선언
link2 = soup.select_one('div#class1 p#second')
print(link2)
print()
internal_link = soup.select_one('div#link a.internal_link')
print(internal_link['href'])
print(internal_link.text)
print()


# select(): find_all()과 같음
# 모든 <h1> 태그 검색 후 리스트 형태로 리턴
h1_all = soup.select('h1')
print('h1_all: ', h1_all)
print()

# 모든 url 링크 검색
url_links = soup.select('a')
for link in url_links:
    print(link['href'])
print()

# <div id="class1"> 내부의 모든 url 검색 <a>
div_urls = soup.select('div#class1 > a')
print(div_urls)     # 인덱스 계산을 잘해야함!!
print(div_urls[0]['href'])
print()

# <div_url2="class1"> 내부의 모든 <a> 태그는 자손 관계
# - 공백으로 구분할 수 있음
div_urls2 = soup.select('div#class1 a')
print(div_urls2)
print()

# 여러 항목 검색하기
# <h1> 태그의 id가 "heading"과 "footer"를 모두 검색
# - ,로 나열
h1 = soup.select('#heading, #footer')
print(h1)
print()

# <a> 태그의 class 이름이 'external_link'와 'internal_link' 모두 검색
url_links = soup.select('a.external_link, a.internal_link')
print(url_links)
