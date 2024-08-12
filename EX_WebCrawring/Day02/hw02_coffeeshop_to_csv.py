# kdt6 황지원
# 전국의 할리스 매장의 위치 트롤링
# 지역, 매장명, 매장 주소, 전화번호
# 수집된 정보는 csv 파일로 저장
# 저장 파일명: hollys_branches.csv

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd # csv 파일로 저장하기 위함

num = [str(x) for x in range(1, 51)]
hollys_df = pd.DataFrame(columns=['매장이름', '지역', '주소', '전화번호'])

for x in num:
    url = f'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo={x}&sido=&gugun=&store='
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    table = soup.select('td')
    region_list = []    # 0
    name_list = []      # 1
    address_list = []   # 3
    tel_num_list = []   # 5

    region_num = [x for x in range(0, 60, 6)]
    name_num = [x for x in range(1, 60, 6)]
    address_num = [x for x in range(3, 60, 6)]
    tel_num = [x for x in range(5, 60, 6)]

    for i in range(len(table)):
        if i in region_num:
            region_list.append(table[i].text)
        if i in name_num:
            name_list.append(table[i].text)
        if i in address_num:
            address_list.append(table[i].text)
        if i in tel_num:
            tel_num_list.append(table[i].text)
            
    sub_hollys_df = pd.DataFrame(columns=['매장이름', '지역', '주소', '전화번호'])
    sub_hollys_df['매장이름'] = name_list
    sub_hollys_df['지역'] = region_list
    sub_hollys_df['주소'] = address_list
    sub_hollys_df['전화번호'] = tel_num_list
    
    hollys_df = pd.concat([hollys_df, sub_hollys_df], ignore_index=True)

for i in range(len(hollys_df.index)):
    print(f"[{i:3}] 매장이름: {hollys_df.loc[i,'매장이름']}, 지역: {hollys_df.loc[i,'지역']}, 주소: {hollys_df.loc[i, '주소']}, 전화번호: {hollys_df.loc[i, '전화번호']}")

print(hollys_df)
hollys_df.to_csv('./hollys_branches.csv', encoding='utf-8', index=False)
print('hollys_branches.csv 파일 저장 완료')
