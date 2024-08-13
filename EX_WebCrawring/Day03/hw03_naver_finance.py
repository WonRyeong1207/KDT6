# kdt6 황지원

# 데이터 크롤링과 정제 : 네이버 증시 정보 크롤링
# 시가 총액 10위 까지의 기업 정보를 크롤링
# - 네이버 금융 웹사이트: https://finance.naver.com/sise/sise_market_sum.naver
# 크롤링 항목 7개 출력
# - 종목명, 종목코드, 현재가, 전일가, 시가, 고가, 저가

# 필요한 라이브러리는 어떤 것이 있을까?
import pandas as pd
from bs4 import BeautifulSoup
import requests

# BeautifulSoup 오류 날 수 있으니
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable


# 코스티지수가 높은 회사명과 주소 가져오는 함수
def cospi(base_url, market_url):
    # 일단은 웹 페이지부터 가져오자
    url = base_url + market_url
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')

    # 코스피? 10개 가져오자
    tbody_list = soup.select('tbody')[0]
    td_list = tbody_list.select('td')

    company_list = []
    company_link_list = []

    for i in range(len(td_list)):
        if td_list[i].find('a', {'class':'tltle'}):
            company_list.append(td_list[i].text)
            company_link_list.append(td_list[i].find('a', {'class':'tltle'}).attrs['href'])

    return company_list, company_link_list

# 상위 10개 항목 정보 뽑아오는 함수
def top_10_company_info(base_url, company_link_list):
    company_name_list = []
    company_code_list = []
    company_pre_price_list = []
    company_last_price_list = []
    company_price_list = []
    company_max_price_list = []
    company_min_price_list = []
    
    for i in range(10):
        url = base_url + company_link_list[i]
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')

        # 정보를 들고오자
        data = soup.find('dl', {'class':'blind'}).text.split()
        
        # 정보를 리스트에 저장
        company_name_list.append(data[11])
        company_code_list.append(data[13])
        company_pre_price_list.append(data[16])
        company_last_price_list.append(data[24])
        company_price_list.append(data[26])
        company_max_price_list.append(data[28])
        company_min_price_list.append(data[30])
    
    # 값을 DF에 저장
    company_info_df = pd.DataFrame(columns=['종목명', '종목코드', '현재가', '전일가', '시가', '고가', '저가'])
    company_info_df['종목명'] = company_name_list
    company_info_df['종목코드'] = company_code_list
    company_info_df['현재가'] = company_pre_price_list
    company_info_df['전일가'] = company_last_price_list
    company_info_df['시가'] = company_price_list
    company_info_df['고가'] = company_max_price_list
    company_info_df['저가'] = company_min_price_list
    company_info_df.index = company_info_df.index + 1
    
    return company_info_df

# 메뉴화면을 보여주는 함수
def menu(company_list):
    print('-'*25)
    print('[ 네이버 코스피 상위 10개 기업 목록 ]')
    print('-'*25)
    for i in range(10):
        print(f"[{i+1}] {company_list[i]}")

# 결과를 보여주는 함수
def print_result(key, company_link, company_df):
    key = int(key)
    print(company_link[key])
    print(f"종목명: {company_df.loc[key, '종목명']}")
    print(f"종목코드: {company_df.loc[key, '종목코드']}")
    print(f"현재가: {company_df.loc[key, '현재가']}")
    print(f"전일가: {company_df.loc[key, '전일가']}")
    print(f"시가: {company_df.loc[key, '시가']}")
    print(f"고가: {company_df.loc[key, '고가']}")
    print(f"저가: {company_df.loc[key, '저가']}")


# 메인 함수
def main():
    base_url = 'https://finance.naver.com'
    market_url = '/sise/sise_market_sum.naver'

    company_list, company_link_list = cospi(base_url, market_url)
    company_df = top_10_company_info(base_url, company_link_list)
    
    while True:
        menu(company_list)
        key = input('주가를 검색할 기업의 번호를 입력하세요(-1: 종료): ')
        
        if key == '-1':
            print('프로그램 종료')
            break
        
        elif int(key) == 1:
            print_result(key, company_link_list, company_df)
        elif int(key) == 2:
            print_result(key, company_link_list, company_df)
        elif int(key) == 3:
            print_result(key, company_link_list, company_df)
        elif int(key) == 4:
            print_result(key, company_link_list, company_df)
        elif int(key) == 5:
            print_result(key, company_link_list, company_df)
        elif int(key) == 6:
            print_result(key, company_link_list, company_df)
        elif int(key) == 7:
            print_result(key, company_link_list, company_df)
        elif int(key) == 8:
            print_result(key, company_link_list, company_df)
        elif int(key) == 9:
            print_result(key, company_link_list, company_df)
        elif int(key) == 10:
            print_result(key, company_link_list, company_df)
        else:
            continue

if __name__ == '__main__':
    
    # print(company_df)
    main()