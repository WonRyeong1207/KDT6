# Q.121
data = input()
# data = 'a'

print(data.upper()) if (data.islower()) else print(data.lower())
print()

# Q.122
score = int(input("score: "))
# score = 79
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

money, m_type = input("입력: ").split()
# money, m_type = '130', '엔'
money = int(money)

print("%.2f 원" % (money * money_rate[m_type]))
print()

# Q.124
num1 = int(input("input Number1: "))
num2 = int(input("input Number2: "))
num3 = int(input("input Number3: "))
# num1 = 10
# num2 = 9
# num3 = 20

print(max(num1, num2, num3))
print()

# Q.125
phone_type = {'011':'SKT',
              '016':'Kt',
              '019':'LGU',
              '010':'알수없음'}

phone, *_ = input("후대전화 번호 입력: ").split('-')
# phone = '010'

if (phone in phone_type.keys()):
    print(f"당신은 {phone_type[phone]} 사용자입니다.")
print()

# Q,126
address = {'010':'강북구',
           '011':'강북구',
           '012':'강북구',
           '013':'도봉구',
           '014':'도봉구',
           '015':'도봉구',
           '016':'노원구',
           '017':'노원구',
           '018':'노원구',
           '019':'노원구'}

address_num = intput("우편번호: ")
# address_num = '01400'

if (address_num[:3] in address.keys()):
    print(address[address_num[:3]])
print()

# Q.127
person_num = input("주민등록번호: ")
# person_num = '990701-1234567'

if ((int(person_num[7])==1) or (int(person_num[7])==3)):
    print("남자")
elif((int(person_num[7])==2) or (int(person_num[7])==4)):
    print("여자")
print()

# Q.128
person_num = input("주민등록번호: ")

if (0 <= int(person_num[8:10]) <= 8):
    print("서울 입니다.")
else:
    print("서울이 아닙니다.")
print()

# Q.129
person_num = input("주민등록번호: ")
# num = 0
num_list = [2, 3, 4, 5, 6, 7, '', 8, 9, 2, 3, 4, 5]
for i in range(len(person_num)-1):
    if (i==6):
        continue
    else:
        num = int(person_num[i]) * num_list[i]
mean = num % 11
total = str(11-mean)

if (total == person_num[-1]):
    print("유효한 주민등록번호입니다.")
else:
    print("유효하지 않은 주민등록번호입니다.")
print()

# Q.130
import requests
btc = requests.get("https://bithumb.com/public/ticker/").json()['data']

price_range = float(btc['max_price']) - float(btc['min_price'])
prince = float(btc['opening_price'])
max_price = float(btc['max_price'])

if ((prince+price_range) > max_price):
    print("상승가")
else:
    print("하락장")
print()

