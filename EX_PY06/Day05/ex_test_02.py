# 조건부표현식

# 실습 : 임의의 숫자가 5의 배수인지 결과를 출력

num = 4
# num = int(input("숫자 : "))

print(f"숫자 {num}은/는 5의 배수 입니다.") if not (num%5) else print(f"숫자 {num}은/는 5의 배수가 아닙니다.")

result = '5의 배수 입니다.' if not (num%5) else '5의 배수가 아닙니다.'
print(f"숫자 {num}은/는 {result}")


print()
# 실습 : 문자열을 입력받아서 문자열 원소 개수를 저장
# - 단 원소의 개수가 0면 None을 저장

# data = input("문자열 : ")
data = 'I loved coffee.'

result = len(data) if len(data) else None
print(f"입력받은 문자열\'{data}\'의 개수는 {result} 입니다.")


print()
# 실습 : 연산자(사칙연산자 : +, -, *, /)와 숫자 2개 입력 받기
#  - 입력된 연산자에 따라 계산 결과 저장

op, n1, n2 = input("사칙연산자와 숫자 2개 입력(예 : + 10 3) : ").split()
try:
    n1, n2 = int(n1), int(n2)
    
    if (op == '+'):
        result = n1 + n2
    elif (op == '-'):
        result = n1 - n2
    elif (op == '*'):
        result = n1 * n2
    elif (op == '/'):
        result = n1 / n2
    else:
        print("연산자가 아닙니다.")
        result = None
    
    print(f"첫번째 숫자 {n1}과 두번째 숫자 {n2}와 연산자 {op}의 결과 : {result}")
except:
    print("숫자가 아닙니다.")
