# 제어문

# 실습
# 메세지를 입력받음
# 알파벳 대문자인 경우 소문자로, 소문자인 경우 대문자로 나머지는 그대로 출력

msg = input("msg : ")
msg2 = ''

for i in msg:
    if ('a' <= i <= 'z'):
        # msg2 = msg2 + i.upper()
        msg2 = msg2 + chr((ord(i) - 32)) #ord() : 문자를 --> 코드값, chr() : 코드값 --> 문자열
    elif ('A' <= i <= 'Z'):
        # msg2 = msg2 +i.lower()
        msg2 = msg2 + chr(ord(i) + 32)
    else:
        msg2 = msg2 + i
print(msg2)

