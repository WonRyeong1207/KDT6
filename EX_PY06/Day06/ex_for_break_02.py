# 제어문 - 반복문 break
# - 중첩 반복문일 경우의 break는 가장 가까이의 있는 반복문만 종료

# 실습 : 단의 숫자 만큼만 구구단 출력
# 2 * 1 = 2  2 * 2 = 4
# 3 * 1 = 3  3 * 2 = 6  3 * 3 = 9

dan = int(input("출력 원하는 단 입력: "))
# isBreak = False # 내부 for문이 종료되면 밖도 같이 종료

for n in range(2, 10):
    print(f"\n[{n}] 단", end=' ')
    for i in range(1, 10):
        print(f"{n} * {i} = {n*i:2}", end='   ') # {xx:<>n} 안의 값 xx를 n자리까지 왼/오로 정렬
        if (i == n):
            # isBreak = True
            break
    # if isBreak: break
    if (n == dan): break
print()

for n in range(2, dan+1):
    print(f"\n[{n}] 단", end='  ')
    for i in range(1, n+1):
        print(f"{n} * {i} = {n*i:2}", end='    ')
print()

