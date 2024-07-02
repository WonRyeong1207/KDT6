# Q.121
# data = input
data = 'a'

print(data.upper()) if (data.islower()) else print(data.lower())
print()

# Q.122
# score = int(input("score: "))
score = 79
grade = ''

if (80 < score):
    grade = 'A'
elif (60 < score):
    grade = 'B'
elif (40 < score):
    grade = 'C'
elif (20 , score):
    grade = 'D'
else:
    grade = 'E'
print(f"grade is {grade}")
print()

# Q.123
money_rate = {'달러':1167,
              '엔':1.096,
              '유로':1268,
              '위안':171}

# money, m_type = input("입력: ").split()
money, m_type = '130', '엔'
money = int(money)

print("%.2f 원" % (money * money_rate[m_type]))
print()

# Q.124
# num1 = int(input("input Number1: "))
# num2 = int(input("input Number2: "))
# num3 = int(input("input Number3: "))
num1 = 10
num2 = 9
num3 = 20

print(max(num1, num2, num3))
print()

# Q.125
phone_type = {'011':'SKT',
              '016':'Kt',
              '019':'LGU',
              '010':'알수없음'}
# phone, *_ = input("후대전화 번호 입력: ").split('-')
phone = '010'

if (phone in phone_type.keys()):
    print(f"당신은 {phone_type[phone]} 사용자입니다.")
print()

