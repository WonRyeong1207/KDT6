# 초기값이 존재하는 매개변수를 사용하느 함수
# - def 함수명(매개변수1, 매개변수2, 매개변수3=초기값)

def add(num1=0, num2=0): # 함수 호출시 데이터가 전달되지 않는 경우 미리 지정된 값으로 처리
    return num1 + num2

print(add())
print(add(4))
print(add(4, 5))

# 함수 기능 : 회원가입
# 함수 이름 : register
# 매개 변수 : id, pw, gender = man
# 함수 결과 : xxx님 가입을 환영합니다 -> str

def register(_id, pw, gender='man'): # 초기값은 맨 마지막에 존재해야함. 순서대로 값을 넣기 때문
    return f"{_id}-{gender}님 가입을 환영합니다."

print(register('hadesdis', 123456789))
print(register('hadesdis', 123456789, 'woman'))

def test(n1, *num, n2=2): # 초기값을 주지 않는 이상 *은 앞에 올 수 없음
    print(n1, n2, num)
test(1, 2, 3, 3, 4, 2)
# def test(n1, n2, *num): 이 젤 베스트