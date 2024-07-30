# 파이썬 과제  2

# 딕셔너리 생성 및 정렬 프로그램
# - 아래의 주어진 총 6개 나라의 수도에 대한 국가명, 대륙, 인구수를 표시한 테이블을 이용하여
#   딕셔너리를 작성하고, 아래 실행 화면과 같이 출력을 하는 프로그램을 작성하시오.

'''
수도 이름(key)    국가명    대륙        인구수
Seoul    South Korea    Asia        9,655,000
Tokyo       Japan       Asia        14,110,000
Beijing     China       Asia        21,540,000
London      United Kingdon      Europe      14,800,000
Berlin      Germany     Europe      3,426,000
Mexico City     Mexico      America     21,200,000
'''

# while을 이용한 구조 : 개인 파이썬 플젝과 유사
# [기능 구현 내용]
# 1. 전체 데이터 출력
#   - 위의 테이블을 이욜하여 dictionary를 생성하고 전체 데이터를 화면세 출력
#   - Key: 수도 이름, value: 국가명, 대륙, 인구수
# 2. 수도 이음 오름차순 출력
#   - 수도 이름을 기준으로 오름차순 정렬한 다음 dictionary의 모든 데이터를 출력
#   - 자리 수 맞춤
# 3. 모든 도시의 인구수 내림차순 출력
#   - 인구수를 내림차순으로 정렬한 다음 수도이름, 인구수만 화면에 출력
#   - 자리 수 맞춤
# 4. 특정 도시의 정보 출력
#   - 화면에서 입력받은 수도 이름이 딕셔너리의 key에 존재하면, 해당 수도의 모든 정보를 화면에 출력함.
#   - 수도 이름이 딕셔너리에 존재하지 않으면 "도시이름: XXX은 key에 없습니다."를 출력
# 5. 대륙별 인구수 계산 및 출력
#   - 화면에서 대륙 이름을 입력 받고 해당 대륙에 속한 국가들의 인구스를 출력하고 전체 인구수 합을 계산하여 출력
#   - 잘못된 대룩 이름 검사는 없음
# 6. 프로그램 종료

# 딕셔너리
contry_dict = {'Seoul':['South Korea', 'Asia', 9655000],
               'Tokyo':['Japan', 'Asia',14110000],
               'Beijing':['China', 'Asia', 21540000],
               'London':['United Kingdom', 'Europe', 14800000],
               'Berlin':['Germany', 'Europe', 3426000],
               'Mexico City':['Mexico', 'America', 21200000]}

# 계속 나오는 것
keys_list = list(contry_dict.keys())
values_list = list(contry_dict.values())

# 화면을 띄우는 함수
def back_graund():
    print('-' * 80)
    print('1. 전체 데이터 출력')
    print('2. 수도 이름 오름차순 출력')
    print('3. 모든 도시의 인구수 내림차순 출력')
    print('4. 특정 도시의 정보 출력')
    print('5. 대룩별 인구수 계산 및 출력')
    print('6.프로그램 종료')
    print('-' * 80)
    
# test
# back_graund()

# 전체 데이터를 출력하는 함수
def all_data_show():
    for i in range(len(keys_list)):
        print(f"[{i+1}] {keys_list[i]}: {values_list[i]}")
    
# test
# all_data_show()

# 수도 이름 오름차순 정렬을 보여주느 ㄴ함수
def capital_sorted():
    sorted_key_list = sorted(keys_list)
    sorted_item_list = sorted(contry_dict.items())
    for i in range(len(keys_list)):
        print(f"[{i+1}] {sorted_key_list[i]}\t: {sorted_item_list[i][1][0]:15} {sorted_item_list[i][1][1]:8} {sorted_item_list[i][1][2]:12,}")
        
# test
# capital_sorted()

# 모든 도시의 인구 수를 내림차순
def num_sorted():
    carry = []
    for i in range(len(values_list)):
        carry.append(values_list[i][2])
    
    contry_popul_dict = dict(zip(keys_list, carry))
    x = sorted(contry_popul_dict.items(), key=(lambda items: items[1]), reverse=True)
    # 람다 머리 쪼개질것 같다
    for i in range(len(keys_list)):
        print(f"[{i+1}] {x[i][0]}\t: {x[i][1]:12,}")

# test
# num_sorted()

# 특정 도시의 정보 출력
def city_info():
    city_name = input("출력할 도시 이름을 입력하세요: ")
    if city_name in keys_list:
        city_info_list = contry_dict[city_name]
        print(f"도시: {city_name}")
        print(f"국가: {city_info_list[0]}, 대륙: {city_info_list[1]}, 인구수: {city_info_list[2]:,}")
    else:
        print(f"도시이름: {city_name}은 key에 없습니다.")
        
# test
# city_info()

# 대륙별 인구수 계산 및 출력
def sum_popul():
    continent = input("대륙 이름을 입력하세요 (Aisa, Europe, America): ")
    continent_total = 0
    value = []
    key = []
    
    for i in range(len(keys_list)):
        if (values_list[i][1] == continent):
            key.append(keys_list[i])
            value.append(values_list[i][2])
            print(f"{key[i]}: {value[i]:,}")
            
    continent_total = sum(value)
    print(f"{continent} 전체 인구수: {continent_total:,}")
    
# test
# sum_popul()

# 입력 데이터 유효성 체크
def input_check(key):
    if len(key) == 1:
        if key.isdecimal():
            return True
        else:
            return False
    else:
        return False

# 입력을 받는 함수
def input_key():
    key = input("메뉴를 입력하세요: ")
    if input_check(key):
        key = int(key)
        return key
    else:
        return None


# 메인 몸체
while True:
    back_graund()
    n = input_key()
    
    
    if n == 6:
        print("프로그램을 종료합니다.")
        break
    
    elif n == 1:
        all_data_show()
        continue
    
    elif n == 2:
        capital_sorted()
        continue
    
    elif n == 3:
        num_sorted()
        continue
    
    elif n == 4:
        city_info()
        continue
    
    elif n == 5:
        sum_popul()
        continue
    
    else:
        continue