# Q.31
a = '3'
b = '4'
print(a + b)
# 34
print('\n')

# Q.32
print("Hi" * 3)
# HiHiHI
print('\n')

# Q.33
print('-' * 80)
print('\n')

# Q.34
t1 = 'python'
t2 = 'java'
print((t1 + t2) * 4)
print('\n')

# Q.35
name1 = '김민수'
age1 = 10
name2 = '이철희'
age2 = 13
print("이름: %s 나이: %d" % (name1, age1))
print("이름: %s 나이: %d" % (name2, age2))
print('\n')

# Q.36
print("이름: {} 나이: {}".format(name1, age1))
print("이름: {} 나이: {}".format(name2, age2))
print('\n')

# Q.37
print(f"이름: {name1} 나이: {age1}")
print(f"이름: {name2} 나이: {age2}")
print('\n')

# Q.38
상장주식수 = '5,969,782,550'
컴마제거 = 상장주식수.replace(',', "")
타입변환 = int(컴마제거)
print(타입변환, type(타입변환))
print('\n')

# Q.39
분기 = '2020/03(E) (IFRS연결)'
print(분기[:7])
print('\n')

# Q.40
data = '    삼성전자    '
data = data.strip()
print(data)
