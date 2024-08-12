# kdt6 황지원
# 특정 지역에 있는 커피 매장 출력하기
import pandas as pd
from tabulate import tabulate

data_df = pd.read_csv('./hollys_branches.csv', encoding='utf-8')
# print(data_df['주소'])
# print(tabulate(data_df.head(),headers=data_df.columns, tablefmt='pretty'))

# 검색된 매장을 리턴하는 함수
def chack_region(region):
    mask = []
    for i in range(len(data_df.index)):
        key = region in data_df.loc[i, '지역']
        mask.append(key)
    region_df = data_df[mask].reset_index(drop='index')
    region_df = region_df.drop('지역', axis=1)
    region_df.index = region_df.index + 1
    return region_df


# 무한 반복
while True:
    region = input('검색할 매장의 지역을 입력하세요: ')
    
    # 정지 조건
    if region == 'quit':
        print('종료 합니다.')
        break
    
    else:
        region_df = chack_region(region)
        
        if len(region_df) == 0:
            print('검색된 매장이 없습니다.')
        else:
            print(f"검색된 매장 수: {len(region_df.index)}")
            print(tabulate(region_df, headers=region_df.columns, tablefmt='pretty'))
            