# 4번. 아래 조건을 만족하는 코드를 작성하세요.
# 조건
# - 수의 범위 : 1 ~ 100
# - 3의 배수 숫자
# - 7의 배수 숫자
# - 8의 배수 숫자
# - 3, 7, 8의 배수 숫자로 구성된 숫자만 출력
# - 단!! 중복된 숫자는 제거 하세요.

num = int(input("수를 입력하세요 : "))

if ((num < 1) or (num > 100)):
    print("수의 범위는 1 ~ 100 입니다.")
    num = int(input("수를 입력하세요 : "))
    
num_3 = []
num_7 = []
num_8 = []

for i in range(1, num+1):
    if (i%3 == 0):
        num_3.append(i)
    if (i%7 == 0):
        num_7.append(i)
    if (i%8 == 0):
        num_8.append(i)

num_3_7_8 = set(num_3+num_7+num_8)
num_3_7_8 = list(num_3_7_8)
num_3_7_8.sort()

print(f"3의 배수 : {num_3}")
print(f"7의 배수 : {num_7}")
print(f"8의 배수 : {num_8}")
print(f"3, 7, 8의 배수로만 구성된 수 : {num_3_7_8}")