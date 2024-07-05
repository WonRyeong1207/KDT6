# 덧셈, 뺄셈, 곱셈, 나눗셈 함수를 각각 만들기
# - 매개 변수 : 정수 2개, num1, num2
# - 함수 결과 : 연산 결과 반환

def add(num1, num2):
    return num1 + num2

def minus(num1, num2):
    return num1 - num2

def multi(num1, num2):
    return num1 * num2

def divi(num1, num2):
    if num2 == 0:
        return None
    else:
        return num1 / num2

print(add(1, 2))
print(minus(1, 2))
print(multi(1, 2))
print(divi(1, 0))

# 한번에 값을 받기
def calu(num1, num2):
    r1 = num1 + num2
    r2 = num1 - num2
    r3 = num1 * num2
    if num2 == 0:
        r4 = None
    else:
        r4 = num1 / num2
    return r1, r2, r3, r4

r1, r2, r3, r4 = calu(100, 5)
print(r1, r2, r3, r4)
print('\n\n')

# 실습 : 사용자로부터 연산자,  숫자1, 숫자2를 입력 받아서 연산결과 출력
# -input().split(',')

def calcu(op, num1, num2):
    if op == '+':
        result = f"{num1} {op} {num2} = {num1 + num2}"
    elif op == '-':
        result = f"{num1} {op} {num2} = {num1 - num2}"
    elif op == '*':
        result = f"{num1} {op} {num2} = {num1 * num2}"
    elif op == '/':
        if num2 == 0:
            result = '0 is not division'
        else:
            result = f"{num1} {op} {num2} = {num1 / num2}"
    else:
        result = f'{op} is not operator'
    
    return result

# 합수 기능: 입력 데이터가 유효한 데이터인지 검사해주는 기능
# 함수 이름 : check_data
# 매개 변수 : 문자열 데이터, 데이터 개수 data, cnt, sep=' '
# 함수 결과 : 유효 여부 T/F

def check_data(data, cnt, sep=' '):
    if len(data):
        data_list = data.split(sep)
        if cnt == len(data_list):
            return True
        else:
            False
    else:
        return False


# 값을 전부 원하는 형태인지 전부 체크 해야함.
data = input("연산자, 숫자1, 숫자2 입력 : ")
# 입력이 있을때 3개가 다 입력되면 쪼개기
if (check_data(data, 3)):
    op, num1, num2 = data.split()

    if (num1.isdecimal() and num2.isdecimal()):
        num1, num2 = int(num1), int(num2)
        print(calcu(op, num1, num2))

    else:
        print('num1, num2 are not intager number.')

else:
    print('data is empty or large data input')
    
