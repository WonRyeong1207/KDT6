# 조건부표현식 : 조건이 2개 이상인 경우

# 실습 : 숫자가 양수, 영, 음수 판별

num = 9
if (num == 0):
    print(f"{num} is zero")
elif (num > 0):
    print(f"{num} is positive")
else:
    print(f"{num} is negative")
    
result = 'positive' if (num > 0) else ('negative' if (num < 0) else 'zero')
print(result)

