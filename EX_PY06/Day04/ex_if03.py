# if 조건문
# 중첩 조건문
# - 형식 if (...):
#        ----if (...):
#            ----xxxxx

# 실습 : 숫자가 음이 아닌 정수와 음수 구분하기
# - 음이 아닌 정수 중에 0과 양수 구분하기

n = int(input("수를 입력 : "))

if (n >= 0):
    if (n == 0):
        print(f"숫자 {n}은 0 입니다.")
    else:
        print(f"숫자 {n}은 양수 입니다.")
else:
    print(f"숫자 {n}은 음수 입니다.")
    
# 동네이름 데이터에서 입력받은 동네이름 해당 부여
city = ['Deagu', 'Busan', 'Ulsan', 'Gangjoo', 'Deajeon']
data = 'Masan'

if (data in city):
    print(f"{data}는 광역시입니다.")

else:
    print(f"{data}는 광역시다 아닙니다.")