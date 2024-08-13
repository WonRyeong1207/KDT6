# 테이블 데이터를 CSV로 저장

import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from html_table_parser import parser_functions as parse

# 그 BeutifulSoup 불러오기 오류나서...
import collections
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

html = urlopen('https://en.wikipedia.org/wiki/Comparison_of_text_editors')
bs = BeautifulSoup(html, 'html.parser')


table = bs.find_all('table', {'class':'wikitable'})[0]
table_data = parse.make2d(table)    # 2차원 리스트 현태로 변환

# 테이블의 2행을 출력
print('[0]:', table_data[0])
print('[1]:', table_data[1])

# Pandas DataFrame으로 저장 (2행부터 데이터 저장, 1행은 column 이름 사용)
df = pd.DataFrame(table_data[2:], columns=table_data[1])
print(df.head())

# csv 파일로 저장
csvFile = open('./data/editors1.csv', 'w', encoding='utf-8')
writer = csv.writer(csvFile)

for row in table_data:
    writer.writerow(row)

csvFile.close()
