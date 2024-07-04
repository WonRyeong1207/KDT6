# 모듈 : 변수 함수, 클래스가 들어있는 파이썬 파일
# 패키지 : 동일한 목적의 모듈을 모은 것
#        여러개의 모듈 파일들 존재
# 사용법 : import 모듈명/파일명 (단, 확장자 제외)

import random

# 임의의 숫자를 추출
# 임의의 숫자 10개 생성
for i in range(10):
    num = random.random()
    print(int(num*10)) 
print()

# random() : [a, b), randint() : [a, b]
for i in range(10):
    num = random.randint(1, 6)
    print(num)
print()

# 실습 : 로또 프로그램 만들기
# - 1~45 중 중복되지 않는 6개의 수 추출
num_list = []
check_list = []
i = 0

while True:
    num = random.randint(1, 45)
    check_list.append(num) # 나도 모르게 너무 당연하게 append 쓰고 있어서 뭐가 문제인지 몰랐다..ㅋㅋㅋ
    if (i == 0):
        i = i + 1
    elif (check_list[i-1] == check_list[i]):
        del check_list[i]
        i = i - 1
    else:
        num_list.append(num)
        i = i + 1
    if (len(num_list) > 6):
        del num_list[-1]
        num_list = sorted(num_list)
        break
print(f"로또번호 6개 : {num_list}")
print()

# set.union()을 사용하면 set으로 자동으로 중복 삭제 가능
num_set = set()
while len(num_set) < 6:
    num = random.randint(1, 45)
    num_set = num_set.union(set([num]))
print(sorted(num_set))

# set.add()를 사용하면 다른 형식의 원소도 넣을 수 있음. 형변환 X
num_set = set()
while len(num_set) < 6:
    num = random.randint(1, 45)
    num_set.add(num)
print(sorted(num_set))


