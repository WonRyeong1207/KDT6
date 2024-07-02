# 6번. 아래와 같이 출력된 함수를 구현해 주세요.
# 조건
# - 입력 : 출력 원하는 단
# 기능
# - 리스트 커프리헨션(List Conprehension)으로 구현
# 예시
# 입력 : 단 : 3
# 출력 : ----------------- 3단 -----------------
#        3 * 1 =  3    3 * 2 =  6    3 * 3 =  9
#        3 * 4 = 12    3 * 5 = 15    3 * 6 = 18
#        3 * 7 = 21    3 * 8 = 24    3 * 9 = 27

num = int(input("단 : "))
num_list = [x*num for x in range(1, 10)]

print("{0:-^30}".format(f" {num}단 "))
for i in range(1, 10, 3):
    print("{0:^30}".format(f"{num} * {i} = {num*i}  {num} * {i+1} = {num*(i+1)}  {num} * {i+2} = {num*(i+2)}"))
    # print("{0:^30}.format(f"{num} * {i} = {num_list[i]}  [num] * {i+1} = {num_list[i+1]}  {num} * {i+2} = {num_list[i+2]}")

'''
이렇게 하면 자리를 맞춰서 표현할 수 있음.
for i in range(1, 10, 3):
    print("{0: >10}".format(f"{n} * {i} = {n*i}"), end='    ')
    print("{0: >10}".format(f"{n} * {i+1} = {n*(i+1)}"), end='    ')
    print("{0: >10}".format(f"{n} * {i+2} = {n*(i+2)}"))
'''