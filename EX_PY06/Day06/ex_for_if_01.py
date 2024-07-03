# 제어문과 반복문 혼합

# 실습
#  - 1~50까지의 데이터가 있음
# - 해당 데이터에서 3의 배수는 제곱하고 나머지 숫자는 그대로 더해서 합계를 출력

num = 0
for i in range(1, 51):
    if ((i%3) == 3):
        num = num ^ i
    else:
        num = num + i
print(f"1 ~ 50까지 합 (단,3의 배수는 제곱) : {num}")

# 실습
# 메세지에서 알파벳과 숫자를 구분해서 처리합니다
# 알파벳 ★, 숫자는 ♡로 변경해서 출력

msg = "Good 2024"
msg2 =''

for i in msg:
    if (('a' <= i <= 'z') or ('A' <= i <= 'Z')):
        msg2 =  msg2 + "★"
    elif ('0' <= i <= '9'):
        msg2 = msg2 + "♡"
    else:
        msg2 = msg2 + i
print(msg2)
print()

