# list/set/dict와 반복문, 조건문 표현식을 합친 표현
#  - 메모리 사용량 감소 & 속도 빠름
# - 파이썬 겉멋

# 실습 : A 리스트의 데이터를 B 리스트에 담기
#       단, A 리스트에서 짝수 값은 3을 곱하고, 홀수 값은 그대로해서 B 리스트에 담기

a = [1, 2, 3, 4, 5, 6]
b = []
for i in a:
    if (i%2 == 0):
        b.append(i*3)
    else:
        b.append(i)
print(f"{a} and {b}\n")

# c 에 a 데이터 불러오기
c = [x for x in a]
print(c)

# 짝수 데이터만 담기
c = [x*3 for x in a if not (x%2)]
print(c)

# 짝수 데이터는 3을 곱하고 홀수 데이터는 그대로 담기
c = [(x*3 if not (x%2) else x) for x in a] # 모든 원소를 다 담기 위함
print(f"a : {a}\nb : {b}\nc : {c}")
