# 연산자 실습

# 실습 1 : 문자열 데이터 2개에 대한 논리 연산 수행 후 출력
#         비교연산자 출력

data1 = 'Hello'; data2 = 'hello'

print(f"({data1} > {data2}) and ({data1} == {data2}) : {(data1 > data2) and (data1 == data2)}")
print(f"({data1} > {data2}) or ({data1} == {data2}) : {(data1 > data2) or (data1 == data2)}")

# 실습 2 : 정수 1개와 문자열 1개에 대한 논리 연산 후 출력
#         논리 연산은 not만 사용

num = 4.3; msg = 'ame'

print(f"not {num} : {not num}")
print(f"not {msg} : {not msg}")

num = 0; msg=''
print(f"not {num} : {not num}")
print(f"not {msg} : {not msg}")
