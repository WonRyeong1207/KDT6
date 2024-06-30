# 3번. 아래 조건을 만족하는 코드를 작성하세요.
# 조건
# - 'q', 'Q' 입력 전까지 동작
# - 대문자 Q 제외한 나머지 알파벳 입력 시 ♠ 출력
# - 소문자 q 제외한 나머지 알파벳 입력 시 ♤ 출력
# - 0 ~ 9 숫자 입력시 숫자만큼의 ◎ 출력

while(1):
    t = input("입력 : ")
    num = 0
    try:
        num = int(t)
        print("◎"*num)
    except:
        if ((t.islower() == True) and (t != 'q')):
            print("♤")
        if ((t.islower() == False) and (t != 'Q')):
            print("♠")
        if (t == 'q'):
            break
        if (t == 'Q'):
            break