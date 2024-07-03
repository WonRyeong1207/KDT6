# 실습 : 구구단 3단 출력 단, while 사용

dan = 3
i = 1

while (i < 10):
    print(f"{dan} * {i} = {dan*i:2}", end='\t')
    i = i + 1
print('\n')

# 실습 : 1~30 범위의 수 중에서 홀수만 출력

i = 1
while (i <= 30):
    if ((i%2) != 1):
        i += 1
        continue 
    print(i, end=' ')
    i += 1 # 오류 날까봐 잘 안쓰는디 편하긴함
print('\n')

i = 1
while (i <= 30):
    if ((i%2) != 0):
        i += 1
        continue 
    print(i, end=' ')
    i += 1 # 오류 날까봐 잘 안쓰는디 편하긴함
print('\n')
