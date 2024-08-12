# 네이버 블로그 검색

# 검색어: ChatGPT
# 첫 번째 블로그의 타이틀 부분
# <a href="https://blog.naver.com/dmsdud0395..." class="title_link">
# - 클래스 속성에 여러 이름이 존재하는 경우, dot(.)으로 접근 가능
# - select(a.title_link)

# 검색해서 자료를 수집할때 이용할 수 있음
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote  # query에 한글 검색을 하기 위한 방법

# query = 'ChatGTP'
# query = quote('심신안정에 좋은 것')
query = quote('커피')
url	=	f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={query}'
# response = requests.get(url)
# soup = BeatifulSoup(requests.text, 'html.parser')

html = urlopen(url)
soup = BeautifulSoup(html.read(), 'html.parser')
blog_results = soup.select('a.title_link')  # 검색 결과 타이틀
print('검색 결과수: ', len(blog_results))
search_count = len(blog_results)
desc_results = soup.select('a.dsc_link')    # 검색 결과의 간단한 설명

for i in range(search_count):
    title = blog_results[i].text
    link = blog_results[i]['href']
    print(f"{title}, [{link}]")
    print(desc_results[i].text)
    print('-'*80)

# 나중에 워드 클라우드에서 빈도수 결과 볼 수 있음
