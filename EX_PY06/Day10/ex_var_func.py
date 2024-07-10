# 함수와 변수 - 지역/전역 변수

# 전역변수 (global)
# - 파일(py) 내에 존재하며 모든 곳에서 사용가능
# - 프로그램 실행 시 메모리 존재
# - 프로그램 종료 시 메모리에서 삭제

name = 'hong'
total = 100

# 지역변수 (local)
# - 함수 안에 존재하며 함수에서만 사용가능
# - 함수 실햄시 메모리에 존재
# - 함수 종료시 메모리에서 삭제

# 함수기능 : 정수를 덧셈한 후 결과를 반환하는 함수
# 함수이름 : addInt
# 매개변수 : 0개 ~ n개 가변인자 *nums
# 함수결과 : 정수 result
# History : 누가 언제 왜 만들었음?

def addInt(*nums):
    total = 0
    for n in nums:
        total = total + n
    return total

def multiInt(*nums):
    # global total # 없으면 이렇게 불러 오던가
    total1 = 1 # 새롭게 선언하던가
    for n in nums:
        total1 = total1 * n
        # 함수 안에 선언 되어 있는 것이 없기에 함수 밖에서 들고옴
    return total1 + total

def multiInt2(*nums):
    global total # 없으면 이렇게 불러 오던가
    # total1 = 1 # 새롭게 선언하던가
    for n in nums:
        total = total * n # 전역변수의 값을 변경할 경우 그냥 사용이 안됨.
        # 함수 안에 선언 되어 있는 것이 없기에 함수 밖에서 들고옴
    return total

result1 = addInt(1)
print(f"result1 => {result1}")

result2 = multiInt(5)
print(f"result2 => {result2}")

print(f"전 : total => {total}")
result3 = multiInt2(5)
print(f"result3 => {result3}")
print(f"후 : total => {total}")

# print(f"nums => {nums}") # nums는 지역함수라 전역적으로 사용 불가

