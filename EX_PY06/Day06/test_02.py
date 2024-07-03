# 실습 : 10번 숫자 데이터를 입력 받습니다.
# - 숫자 데이터를 모두 더해서 합계가 30이상이면 break

con = 0
for i in range(10):
    num = int(input("숫자: "))
    con = con + num
    print(f"현재 합계는 {con}입니다.")
    if (30 <= con):
        print(f"\n합계는 {con}으로 30이상이 되어 종료합니다.")
        break
    
