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

# input을 확인하는 함수
def check_input(input_list):
    if (len(input_list)==3):
        if (input_list[0] in ['+', '-', '*', '/']):
            # '0'의 예외처리가 필요함.  그래서 쪼갬
            op, num1, num2 = input_list[0], input_list[1], input_list[2]
            
            if (num1.isdecimal() and num2.isdecimal()):
                return op, int(num1), int(num2)
            else:
                print("정수를 입력하지 않으셨습니다.\n다시 입력해주세요.")
                return None, None, None
        else:
            print("올바른 연산자가 아닙니다.\n다시 입력해주세요.")
            return None, None, None
    else:
        print("적거나 많이 입력하였습니다.\n다시 입력해주세요.")
        return None, None, None

# input을 하는 함수
def input_calu():
    print("4칙 연산을 하는 계산기입니다.")
    print("연산자와 정수 2개를 입력해주세요.")
    input_list = input("연산자 정수1 정수2 : ").split()
    print()
    op, num1, num2 = check_input(input_list)
    return op, num1, num2

# 계산기
def calu():
    op , num1, num2 = input_calu()
    
    if ((op==None) and (num1==None) and (num2==None)):
        print("프로그램을 다시 실행시켜주세요.")
    else:
        if (op == '+'):
            result = add(num1, num2)
        elif (op == '-'):
            result = minus(num1, num2)
        elif (op == '*'):
            result = multi(num1, num2)
        else:
            result = div(num1, num2)
        
        print(f"입력하신 연산자와 정수의 결과입니다.")
        print(f"{num1} {op} {num2} = {result}\n")
    

calu()
