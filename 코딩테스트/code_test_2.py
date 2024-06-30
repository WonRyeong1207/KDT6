# 2번. 입력 받은 데이터 중에서 숫자만 모두 저장하여 합계, 최대값, 최소값을 출력하는 코드를 구현하세요.
# 예시
# 입력 : 데이터 입력 : 하늘 Apple 2021 9 False 23 7 None 끝
# 출력 : 합계 : 2060    최댓값 : 2021    최솟값 : 7

data = input("데이터 입력 : ").split()


num_list =[]
for i in range(len(data)):
    num = []
    current = data[i]
    for j in current:
        try:
            int(j)
            num.append(j)
        except:
            continue
    if (len(num) != 0):
        num = ''.join(num)
        num_list.append(int(num))
    # print(num_list)

num_sum = sum(num_list)
num_max = max(num_list)
num_min = min(num_list)
print(f"합계 : {num_sum},    최댓값 : {num_max},    최솟값 : {num_min}")

if (num_list == None):
    print("합계 : 0,    최댓값 : 0,    최솟값 : 0")
