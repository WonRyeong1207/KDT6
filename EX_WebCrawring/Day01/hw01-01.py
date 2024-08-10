# kdt6 황지원

# 날씨 현황 HTML 분석
# url = 'https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168'

# find_all이나 select로 항목부터 찾아야함

# 출력항목 (4개)
# - <p class="period-name">"Overnight"</p>
# - <p class="short-desc">"Mostly"</p>
# - <p class="temp temp-low">"Low 55F"</p>
# - <img class="..." title="Overnight: ~~~~">

# 2개의 함수 출력
# def scraping_use_find(html)
# - find(), find_all() 사용
# def scraping_use_selsect(html)
# - select(), select_one() 사용

# 필요한 라이브러리
from bs4 import BeautifulSoup
from urllib.request import urlopen


def scraping_use_find(html):
    div_class_list = html.find_all('div', {'class':'tombstone-container'})
    print('[find 함수 사용]')
    print('총 tombstone-container 검색 개수: ', len(div_class_list))
    for div_class in div_class_list:
        period_name = div_class.find('p', {'class':'period-name'}).string
        short_desc = div_class.find('p', {'class':'short-desc'}).string
        temp = div_class.find('p', {'class':'temp'}).string
        img_title = div_class.find('img')['title']
        print('-'*80)
        print(f"[Period]: {period_name}")
        print(f"[Short desc]: {short_desc}")
        print(f"[Temperature]: {temp}")
        print(f"[Image desc]: {img_title}")
    print('-'*80)


def scraping_use_selsect(html):
    div_class_list = html.select('div.tombstone-container')
    print('[select 함수 사용]')
    print('총 tombstone-container 검색 개수: ', len(div_class_list))
    for div_class in div_class_list:
        period_name = div_class.select_one('p.period-name').string
        short_desc = div_class.select_one('p.short-desc').string
        temp = div_class.select_one('p.temp').string
        img_title = div_class.select_one('img')['title']
        print('-'*80)
        print(f"[Period]: {period_name}")
        print(f"[Short desc]: {short_desc}")
        print(f"[Temperature]: {temp}")
        print(f"[Image desc]: {img_title}")
    print('-'*80)




def main():
    url = 'https://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168'
    page = urlopen(url)
    html = BeautifulSoup(page.read(), 'html.parser')
    
    scraping_use_find(html)
    scraping_use_selsect(html)
    
main()