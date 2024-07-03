# 실습 : 숫자를 입력받아 양수와 음수 구분하기

num = int(input("수를 입력하세요 : "))

if (num == 0):
    print(f"숫자 {num}은 0 입니다.")
elif (num < 0):
    print(f"숫자 {num}은 음수 입니다.")
else:
    print(f"숫자 {num}은 양수 입니다.")
    
print('\n')

# 실습 : 점수를 입력 받아 합격과 불합격 출력
# - 합격 : 60점 이상

score = int(input("점수를 입력하세요 : "))
if (score >= 60):
    print("합격\n")

# 실습 : 점수를 입력받아서 학점 출력
# - 학점 : A(90이상), B(80이상), C(70이상), D(60이상), F

grade = int(input("점수를 입력하세요 : "))

if (grade >= 90):
    print('A')
elif (grade >= 80):
    print('B')
elif (grade >= 70):
    print('C')
elif (grade >= 60) :
    print('D')
else:
    print('E')
    
    