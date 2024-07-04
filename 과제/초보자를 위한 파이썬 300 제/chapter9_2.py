# Q.211
def 함수(문자열):
    print(문자열)
    
함수("안녕")
함수("Hi")
# 안녕\nHi
print()

# Q.212
def 함수(a, b):
    print(a + b)

함수(3, 4)
함수(7, 8)
# 7\n15
print()

# Q.213
def 함수(문자열):
    print(문자열)
# 함수()
# 함수 안에 문자열을 넣어줘야하는데 안 넣었기 때문에

# Q.214
def 함수(a, b):
    print(a + b)
# 함수("안녕", 3)
# 인자는 문자열과 정수형은 서로 형변환을 하지 않으면 연산을 할 수 없음

# Q.215
def print_with_smile(data):
    print(data+":D")
    
# Q.216
print_with_smile("안녕하세요")
print()

# Q.217
def print_upper_price(price):
    print(price*1.3) # 상한가는 1 + n% 인가...

# Q.218
def print_sum(n1, n2):
    print(n1 + n2)
    
# Q.219
def print_arithmetic_operation(n1, n2):
    print(f"{n1} + {n2} = {n1+n2}")
    print(f"{n1} - {n2} = {n1-n2}")
    print(f"{n1} * {n2} = {n1*n2}")
    print(f"{n1} / {n2} = {n1/n2}")

# Q.220
def print_max(n1, n2, n3):
    c = o
    if (c < n1):
        c = n1
    if (c < n2):
        c = n2
    if (c < n3):
        c = n3
    print(c)
