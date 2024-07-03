# 제어문 - while

# 실습 : 사용자로부터 데이터를 입력받습니다.
#        사용자로부터 'q'나 'Q'를 입력받으면 입력 받기를 중단
#       그전 까지는 계속 입력 받음

while True:
    n = input("문자 : ")
    
    if ((n == 'q') or (n == 'Q')): # whlie은 반드시 조건을 종료 시킬 조건이 필요하다
        break
    print(n)
print()
