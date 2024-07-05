# 사용자 정의 함수
# 함수기능, 함수이름, 매개변수, 함수결과 를 지정해야함

# 함수 기능 : 2개인 정수를 덧셈한 후 결과를 반환
# 함수 이름 : add
# 매개 변수 : 2개, num1, num2
# 함수 결과 : result = num1 + num2

def add(num1, num2):
    
    result = num1 + num2
    
    return result

# 함수 호출
print(add(1, 2))
# print(add(1, 2, 3)) # 매개변수 초과로 인한 오류 발생
# print(add(1)) # 매개변수 부족으로 인한 오류 발생

