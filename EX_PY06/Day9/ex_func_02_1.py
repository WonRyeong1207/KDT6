# 함수 funtion 이해 및 활용
# 함수 기반 계산기 프로그램
# - 4칙 연산 기능별 함수 생성 => 덧셈, 뺄셈, 곱셈, 나눗셈
# - 2개의 정수만 계산

# 4칙 연산 함수
def add(n1, n2):
    result = n1 + n2
    return result

def minus(n1, n2):
    result = n1 - n2
    return result

def multi(n1, n2):
    result = n1 * n2
    return result

def div(n1, n2):
    if n2 == 0:
        result = 'None'
    else:
        result = n1 / n2
    return result

# 계산기 프로그램
# - 사용자가 종료를 원할때 종료 => 'x', 'X' 입력시
# - 연산방식과 숫자 데이터 입력 받기

while True:
    # 입력받기
    req = input("연산(+, -, *, /) 방식과 정수 2개 입력(예: + 10 2) : ")
    
    # 종료조건
    if (req == 'x' or req == 'X'):
        print('계산기를 졸료합니다.')
        break
    
    # 입력에 대한 연산방식과 숫자데이터 추출
    else:
        op, num1, num2 = req.split()
        
        # 숫자는 정수로 변환
        num1 = int(num1)
        num2 = int(num2)
        
        if (op == '+'): print(f"{num1} {op} {num2} = {add(num1, num2)}")
        elif (op == '-'): print(f"{num1} {op} {num2} = {minus(num1, num2)}")
        elif (op == '*'): print(f"{num1} {op} {num2} = {multi(num1, num2)}")
        elif (op == '/'): print(f"{num1} {op} {num2} = {div(num1, num2)}")
        else: print("지원하지 않는 기능 입니다.")
    
    