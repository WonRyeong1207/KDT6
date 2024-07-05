# 가변인자를 받는 함수
# 변수를 함수 선언한 것보다 더 ㅁㅏㄶ이 받을때

# 매개변수의 개수를 유동적으로 0 ~ N까지 가능하도록
# def 함수명(*변수명): <-- 0~ N개의  데이터를 받을 수 있음

# 함수 기능 : 정수를 덧셈 후 반환
# 함수 이름 : add
# 매개 면수 : 0 ~ N
# 함수 결과 : 덧셈 결과 result

def add(*int_num): # 가변 인자 함수
    result = 0
    
    for i in int_num:
        result += i
        
    # result = sum(int_num)
    return result

n = add(1, 2, 3, 4)
print(n)
