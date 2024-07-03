# 문자열 str 데이터 다루기
# 문자열 요소 : 산술, 비교, 멤버 연산

# 산술연산
data1 = 'Happy'
data2 = 'Year'

# str + str : strstr 연결됨
print(f"{data1} + {data2} : {data1+data2}")
print(f"{data1} + {10} : {data1+str(10)}")
# str -* str : 지원하지 않음.
# print(f"{data1} - {data2} : {data1-data2}")
print(f"{data1} * {10} : {data1*10}")


# 멤버 연산
# 요소/원소 in 문자열 : 존재 - T, 미존재 - F
# 요소/원소 not in 문자열 : 존재 - F, 미존재 - T
print(f'h in {data1} : {"h" in data1}')
print(f'h not in {data1} : {"h" not in data1}')
