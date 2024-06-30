# 7번. 숫자와 콤마로만 이루어진 문자열 data가 주어진 이때, data에 포함되어있는
#      자연수의 합과 가장 작은 수, 가장 큰 수를 출력하는 함수를 구현하세요.
# 예시
# 입력 : data = '123,42,98,18'
# 함수호출
# 출력 : "123,42,98,18"의 합 : 38,    가장 큰 수 : 9,    가장 작은 수 : 1

data = input("data = ").split(',')

def SumMaxMin(data_list):
    
    num_list = []
    num = []
    for i in range(len(data_list)):
        num_ca = []
        current = data_list[i]
        for j in current:
            try:
                int(j)
                num_ca.append(j)
                num.append(j)
            except:
                continue
        if (len(num) != 0):
            num_ca = ''.join(num_ca)
            num_list.append(int(num_ca))
    
    num_sum = sum(num_list)
    num_max = max(num)
    num_min = min(num)
    
    print(f"{data_list}의 합 : {num_sum},   가장 큰 수 : {num_max},   가장 작은 수 : {num_min}")
    
SumMaxMin(data)