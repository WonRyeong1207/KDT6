# kdt6 황지원
# Jabplanet '기업랭킹 IT/엡/통신'에서 총 만족도 순위 별로 복지 혜택
# 1위부터 2446위 까지 존재
# 복지 혜택은 각 기업정보 링크 넘어가서 있음.

from bs4 import BeautifulSoup
import requests
import pandas as pd     # 일단은 DF로 저장을 해보자 to_csv

# 그 BeutifulSoup 불러오기 오류나서...
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# base_url = 'https://www.jobplanet.co.kr/companies'
# 1번부터 245번까지 페이지
# for i in range(1, 255):
# page_url = f'?industry_id=700&sort_by=review_survey_total_avg_cache&page={i}'
# total_url = base_url + page_url
# 가져올 정보 : 기업이름, 총 만족도, 평균임금, 기업별 이동 링크
# dt class='us_titb_l3'에서 a를 find하면 기업이름(text), 기업별 이동 링크(.attrs['href']) 가져 올 수 있음
# dl class='content_col2_4'에서 span class='gfvalue', strong class='notranslate'를 find 하면 총 만족도(text)와 평균 임금(text)을 가져올 수 있음

# 랭킹 순으로 데이터를 들고 오는 함수
def rank_company_info():
    # 크롤링에 문제가 있음. 그래서 html을 txt 파일로 저장후 불러옴
    # 상위 500개 기업에 대해서 진행을 할 것임
    
    base_url = 'www.jobplanet.co.kr_companies'
    
    # 자료를 저장할 DataFrame
    rank_df = pd.DataFrame(columns=['기업 이름', '이동 링크', '총 만족도', '평균 임금'])
    
    # 반복문 시작
    for i in range(1, 51):
        # 들고온 자료를 저장할 리스트
        name_list = []
        level_list = []
        mean_wage_list = []
        link_list = []
        
        url = open(f"./html/page/{base_url}_page{i}.txt", encoding='utf-8')
        soup = BeautifulSoup(url, 'html.parser')
        url.close()
        
        # dt class='us_titb_l3'
        dt_list = soup.select('dt.us_titb_l3')
        for data in dt_list:
            a = data.find('a')
            name = a.text
            link = a.attrs['href']
            name_list.append(name)
            link_list.append(link)
        
        # dl class='content_col2_4'
        dl_list = soup.select('dl.content_col2_4')
        for data in dl_list:
            level = data.find('span', {'class':'gfvalue'}).text
            level_list.append(level)
            mean_wage = data.find('strong', {'class':'notranslate'}).text
            mean_wage_list.append(mean_wage)

        # DataFrame을 저장하기 위한 임시 DataFram
        carry_df = pd.DataFrame(columns=['기업 이름', '이동 링크', '총 만족도', '평균 임금'])
        carry_df['기업 이름'] = name_list
        carry_df['이동 링크'] = link_list
        carry_df['총 만족도'] = level_list
        carry_df['평균 임금'] = mean_wage_list
        
        rank_df = pd.concat([rank_df, carry_df], ignore_index=True)
        
    rank_df.to_csv('./company_rank_top500.csv', encoding='utf-8', index=False)
    return rank_df
    
# 기업 정보 - 복지 (benefit) 예시) 네이버 웹툰
# company_bene_url = '/328190/benefits/%EB%84%A4%EC%9D%B4%EB%B2%84%EC%9B%B9%ED%88%B0'
# total_ url = base_url + company_bene_url
# div class='welfare-bullet'을 select로 들고와서 리스트 별로 찾아야함!
# h5 class='welfare-bullet__fit'을 find().text하면 분류명이 나옴
# span class='item-name'을  select()하고 리스트 요소별로 .text하면 될듯?

# 복지 정보를 가져오는 함수
def company_bene(link_list):
    # 기본 upl은 같음
    base_url = 'www.jobplanet.co.kr'
    
    # 값을 저장할 데이터 프레임
    bene_df = pd.DataFrame(columns=['기업 이름', '복지 카테고리', '복지 혜택'])
    
    
    for n in range(500):
        link = link_list[n]
        # 실제 주소와  맞추기 위해서 전처리
        link = link.replace('info','benefits')
        # link : /companies/52192/info/%ED%81%B4%EB%A6%BD%EC%86%8C%ED%94%84%ED%8A%B8?
        link = link.replace('/', '_')
        # link : /companies/52192/benefits/%ED%81%B4%EB%A6%BD%EC%86%8C%ED%94%84%ED%8A%B8?
        link = link[:len(link)-1]
        # link : _companies_52192_benefits_%ED%81%B4%EB%A6%BD%EC%86%8C%ED%94%84%ED%8A%B8?
        url = open(f"./html/link/{base_url}{link}.txt", encoding='utf-8')
        # link : _companies_52192_benefits_%ED%81%B4%EB%A6%BD%EC%86%8C%ED%94%84%ED%8A%B8
        # 실제 주소 :https://www.jobplanet.co.kr/companies/329950/benefits/(%EC%A3%BC)%ED%8C%80%EC%97%98%EB%A6%AC%EC%8B%9C%EC%9B%80
        # ()안의 문자는 다 지운 형태 
        #  - :www.jobplanet.co.kr_companies_394922_benefits_%EC%97%94%EC%97%90%EC%9D%B4%EC%B9%98%EC%97%94%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C.txt
        # print(url)
        soup = BeautifulSoup(url, 'html.parser')
        url.close()
        
        # div class='welfare-bullet'
        # 복지가 없는 회사가 있을 수 있기 때문에 있는 경우와 없는 경우를 구별해야함.
        if soup.find('div',{'class':'welfare-bullet'}):
            print('복지 혜택이 존재')
            
            # <h5 class="welfare-provision__tit">'(주)팔라'의 복지</h5>   3개있어서 그중 2번째 사용
            company_list = soup.find_all('h5', {'class':'welfare-provision__tit'})
            # 에러 발생...
            if len(company_list) > 1:
                company = company_list[1].text
            else:
                company = company_list[0].text
            
            # h5 class='welfare-bullet__tit'
            category_list = []
            h5_list = soup.find_all('h5', {'class':'welfare-bullet__tit'})
            for h5 in h5_list:
                category_list.append(h5.text)
            
            # span class='item-name'
            item_list = []
            span_list = soup.find_all('span',{'class':'item-name'})
            for span in span_list:
                item_list.append(span.text)
            
            bene_df.loc[n, :] = company, category_list, item_list
            
            
        else:
            print('등록된 복지 혜택이 없음')
            bene_df.loc[n, :] = None, None, None

    bene_df.to_csv('./company_bene_df.csv', encoding='utf-8', index=False)
    print(bene_df)
    
    

if __name__ == '__main__':
    rank_df = rank_company_info()
    link_list = rank_df.loc[:,'이동 링크'].to_list()
    # print(link_list)
    company_bene(link_list)
    