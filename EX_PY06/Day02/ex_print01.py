# 내장함수 print() 사용법
# - 모니터 또는 콘솔, 터미널에 출력하는 함수
# - 문법 : print(_1, _2, ...)
#          print()

# 나이, 이름, 성별을 저장하기
age = 24
name = '황지원'
gender = '여'

print(age, name, gender)
print(f"나이 : {age}, 이름 : {name}, 성별 : {gender}")

# 2개의 정수 덧셈 결과 출력하기
num1 = 2
num2 = 9
print(num1+num2)
print(num1, '+', num2, '=', num1+num2)
print("num1 + num2 = %d" % (num1+num2))
print(f"num1 + num2 = {num1+num2}")

# ==> 화면 출력 글자를 만들고 그글안에 특정결과를 출력하는 형식
# 글자 내부에 정수결과 넣기 : '%d'
# 글자 내부에 실수결과 넣기 : '%f'또는 '%.f'(소수점 아래 몇자리만 표시)
# 글자 내부에 글자결과 넣기 : '%s'
print("나이 : %d, 이름 : %s" % (age, name))
print("%d + %d = %d" % (num1, num2, num1+num2))

# 9 / 2 = 4.5
print("%d / %d = %f" % (9, 2, 9/2))
print("%d / %d = %.2f" % (9, 2, 9/2))

# 서식지정자를 이용하지 않는 format
print(f"{9} / {2} = {9/2}")
