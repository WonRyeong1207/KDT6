# 공공데이터 과제 3

# 대구광역시 전체 및 9개 구, 군별 (중구, 동구, 서구, 남구, 북구, 수성구, 달서구, 달성군, 군위군) 남녀
# 비율을 각각의 파이 차트로 구현하세요
# - subplot를 이용하여 5X2 형태의 총 10개의 subplot을 파이 차트로 구현
# - gender.csv 파일 사용

# 실행 결과 (화면 출력)
#   대구광역시 : (남: xx,xxx, 야: x,xxx)
#   대구광역시 중구 : (남: xxx,xxx, 여: x,xxx)

# 필요한 모듈과 라이브러리 import
import csv
import pandas as pd     # 안쓰고 해보고 안되면 쓰자
import matplotlib.pyplot as plt
import numpy as np
import koreanize_matplotlib

# 파일경로
FILE_PATH = '../data/gender.csv'

city_list = ['대구광역시', '대구광역시 중구', '대구광역시 동구', '대구광역시 서구',
             '대구광역시 남구', '대구광역시 북구', '대구광역시 수성구',
             '대구광역시 달서구', '대구광역시 달성군', '대구광역시 군위군']

# 일단은 파일에서 남녀 각각의 데이터를 들고 오자
male_list = []
female_list = []

f = open(FILE_PATH, encoding='euc_kr')
data = csv.reader(f)
next(data)

for row in data:
    for city in city_list:
        if city in row[0]:
            male_num = int(row[104].replace(',', ''))
            male_list.append(male_num)
            female_num = int(row[207].replace(',', ''))
            female_list.append(female_num)
            # print(f"{city} : (남: {male_num:,}, 여: {female_num:,})")
f.close()

# 가져온 데이터 정리
# print(male_list, female_list)
carry_m =[]
carry_f = []
carry_m.append(male_list[0])
carry_f.append(female_list[0])
for i in range(1, len(male_list), 2):
    carry_m.append(male_list[i])
    carry_f.append(female_list[i])
male_list = carry_m
female_list = carry_f
# print(male_list)

population_list = []
for i in range(len(city_list)):
    print(f"{city_list[i]} : (남: {male_list[i]:,}, 여: {female_list[i]:,})")
    population_list.append([male_list[i],female_list[i]])

# 파이 차트 그리기
# print(population_list)
fig = plt.figure(figsize=(7, 8))
axes = fig.subplots(5, 2)
for row in range(5):
    for col in range(2):
        index = row * 2 + col   # 언니들 감사
        axes[row, col].pie(population_list[index], labels=['남', '여'], autopct='%.1f%%', startangle=90)
        axes[row, col].set_title(f"{city_list[index]}", fontsize=10)
fig.suptitle("대구광역시 구별 남녀 인구 비율")
fig.set_tight_layout(True)
plt.show()
