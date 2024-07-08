# 함수 (function) 이해하기

# 함수 기능 : 3개의 정수를 덧셈한 후 결과를 반환하는 함수
# 함수 이름 : add_tree
# 매개 변수 : n1, n2, n3
# 함수 결과 : int, result

def add_tree(n1=0, n2=0, n3=0):
    result = n1 + n2 + n3
    return result

# 함수 기능 : 3개의 정수를 곱셈한 후 결과를 반환
# 함수 이름 : mul_tree
# 매개 변수 : n1, n2, n3
# 함수 결과 : int, reslut

def mul_three(n1=1, n2=1, n3=1):
    result = n1 * n2 * n3
    return result

# 함수 기능 : 2개의 정수를 나눗셈한 후 결과를 출력
# 함수 이름 : div_two
# 매개 변수 : n1, n2
# 함수 결과 : print(), float

def div_two(n1=1, n2=1):
    if (n1 ==0 ) or (n2 == 0):
        print("num1 or num2 is 0. 0 division is None.")
    else:
        result = n1 / n2
        print(f"{n1} / {n2} = {result}")
        
# 함수 호출하기

# 덧셈
add_value = add_tree(1, 23, 4)

# 곱셈
mul_value = mul_three(1, 23, 4)

# 나눗셈
div_two(22, 0)
