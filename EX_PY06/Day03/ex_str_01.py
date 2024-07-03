# 문자열 str 다루기

# 여러줄 문자열 ==> ''' ''' 또는 """ """
# 변수 선언 한해주면 주석처리 느낌
msg = '''
오늘은
비오는
정말 싫은날
'''

print(msg)

# 인덱싱 : 문자열 안에 문자 한개 한개 식별하는 방법
# - 원소/요소 : 문자열 안의 문자 1개
# - 문법 : 변수명[인덱스], 문자열[인덱스], 변수명[시작인덱스(포함):끝인덱스(미포함):증가폭]
# -- 인덱스 종류 : 왼>>오 (0,1,2), 왼<<오 (-3,-2,-1) 이때 넘버는 다르지만 시작은 0부터 디폴트

msg = 'Good Day'
print(msg)
print(msg[3])

msg2 = ' '
print(msg2[0]) # 빈 경우에는 사용불가

# 요소의 개수를 파악해주는 함수
# len() : 요소가 있어야 사용가능
print(f"msg의 길이 : {len(msg)}")

data = "Happy New Year 2025! Good Luck"
print(f"인덱스 범위 : 0 ~ {len(data)}")
print(data[15:20])

a = 'Life is too short, You need Python'
print(a[0:4])
print(a.index('Y'))
print(a[19:])

# 데이터 규칙있게 추출
data = '123456789'

data2 = data[1::2] # 짝수 인뎃스만 추출
data3 = data[::2] # 홀수 인덱스만 추출

print(f"짝수 인덱스만 추출 : {data2}")
print(f"홀수 인덱스만 추출 : {data3}")

