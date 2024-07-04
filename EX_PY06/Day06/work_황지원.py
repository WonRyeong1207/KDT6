# 17장부터 20장까지

# p.202 연습문제
i = 2
j = 5

while ((i<=32) or (j>=1)):
    print(i, j)
    i = i * 2
    j = j - 1
print()

# p.203 심사문제
money = int(input())
# money = 13500

while (money > 0):
    money = money - 1350
    print(money)
print()

# p.211 연습문제
i =  0
while True:
    if ((i%10) != 3):
        i = i + 1
        continue
    if (73 < i):
        break
    print(i, end=' ')
    i = i + 1
print('\n')

# p.212 심사문제
start, stop = map(int, input().split())
# start, stop = 21, 33

i = start

while True:
    if ((i%10) == 3):
        i = i + 1
        continue
    if (stop < i):
        break
    print(i, end=' ')
    i = i + 1
print('\n')

# p.218 연습문제
for i in range(5):
    for j in range(5):
        if (j < i):
            print(' ', end='')
        else:
            print('*', end='')    
    print()
print()

# # p.219 심사문제
n = int(input())
# n = 5
for i in range(n):
    for j in range(n):
        if (i < j):
            print(' ', end='')
    if (i == 0):
        print('*', end='')
    else:           
        print("*" * (i + (i*2)), end='')
    print()
print()

# 이게 원하던 코딩
for i in range(n):
    for j in range(n):
        if (i < j):
            print(' ', end='')      
    print("*" * (1 + (i*2)), end='')
    print()
print()

# p.225 연습문제
for i in range(1, 101):
    if (((i%2) == 0) and ((i%11) == 0)):
        print("FizzBuzz")
    elif ((i%2) == 0):
        print('Fizz')
    elif ((i%11) == 0):
        print('Buzz')
    else:
        print(i)
print()
    
# p.226 심사문제
num1, num2 = mpa(int, input().split())
# num1, num2 = 35, 40

for i in range(num1, num2+1):
    if (((i%5) == 0) and ((i%7) == 0)):
        print("FizzBuzz")
    elif ((i%5) == 0):
        print('Fizz')
    elif ((i%7) == 0):
        print('Buzz')
    else:
        print(i)
print()
