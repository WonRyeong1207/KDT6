# 제어문 중 조건문 살펴보기
# - 조건에 만족하는 경우 즉, Ture가 되면 실행되는 코드와 실행되지 않는 코드를 결정하는 문법
# - 사용법 : if (조건):
# -          ----실행코드
# 들여쓰기 한 것만 실행됨.

# 실습 나이에 따른 버스 요금 출력하기
# - 영유아 : 6세 미만 무료
# - 어린이 : 12세 까지 500원
# - 청소년 : 19세 까지 1000원
# - 어른 : 만 64미만 1700원
# - 노인 : 65세 이상 무료

bus_money = {'child' : 0, 'element':500, 'student':1000, 'adult':1700, 'senior':0}
age = int(input("age : "))
# age  20
age_tag = ''

if (0 < age < 6):
    age_tag = 'child'
elif (6 <= age <13):
    age_tag = 'element'
elif (13 <= age < 20):
    age_tag = 'student'
elif (20 <= age < 65):
    age_tag = 'adult'
else:
    age_tag = 'senior'
    
print(f"나이 {age}세는 버스 요금이 {bus_money[age_tag]}원 입니다.\n")

# 조건 2개 이상인 경우
# - 형식 : if (...):
#          ----xxxxxx
#          elif (...):
#          ----xxxxxx
#          else:
#          ----xxxxxx
# 조건식을 만족하면 다음 조건을 안 봄.

# 강의 내용은 나이에 따라서 print 출력
# 조건을 넣어서 출력
# 다중 조건문 사용
# 나이를 입력받아서 출력
