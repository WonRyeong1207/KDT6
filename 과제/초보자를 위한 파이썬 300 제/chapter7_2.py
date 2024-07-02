# Q.111
msg = input()
# msg = '안녕하세요'
print(msg*2)
print()

# Q.112
num = int(input("숫자를 입력하세요: "))
# num = 30
print(num+10)
print()

# Q.113
num = int(input())
# num = 30
if ((num%2) == 0):
    print('짝수')
else:
    print("홀수")
print()

# Q.114
num = int(input("입력값: "))
# num = 240
print(225) if ((num+20) > 255) else print(num+20)
print()

# Q.115
num = int(input("입력값: "))
# num = 15
print(255) if ((num-15) > 255) else (print(0) if ((num-15) < 0) else print(num-15))
print()

# Q.116
_, date = input("현재시간: ").split(':')
# date = '02'
print('정각입니다.') if ((date == '00') or (date == '0')) else print("정각이 아닙니다.")
print()

# Q.117
fruit = ['사과', '포도', '홍시']

food = input("좋아하는 과일은? ")
# food = '귤'
print("정답입니다.") if (food in fruit) else print("오답입니다.")
print()

# Q.118
wern_investment_list = ["Microsoft", "Google", "Naver", "KaKao", "SAMSUNG", "LG"]

investment = input("투자종목명: ")
# investment = 'KNU'
print("투자 경고 종목입니다.") if (investment in wern_investment_list) else print("투자 경고 종목이 아닙니다.")
print()

# Q.119
fruit = {'봄':'딸기', '여름':'토마토', '가을':'사과'}

season = input("좋아하는 계절은: ")
# season = '겨울'
print("정답입니다.") if (season in fruit.keys()) else print("오답입니다.")
print()

# Q.120
food = input("좋아하는 과일은? ")
# food = '귤'
print("정답입니다.") if (food in fruit.values()) else print("오답입니다.")
print()

