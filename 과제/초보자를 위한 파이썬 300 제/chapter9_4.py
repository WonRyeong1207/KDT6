# Q.231
def n_plus_1 (n):
    result = n + 1
    
n_plus_1(3)
# print(result)
# error 함수 내부의 변수는 밖에서 사용불가

# Q.232
def make_url(data):
    print(f"www.{data}.com")

make_url("naver")
print()

# Q.233
def make_list(str_data):
    data_list = [x for x in str_data]
    return data_list

print(make_list("abcd"))
print()

# Q.234
def pickup_even(int_list_data):
    even_list = [x for x in int_list_data if ((x%2)==0)]
    return even_list

print(pickup_even([3, 4, 5, 6, 7, 8]))
print()

# Q.235
def convert_int(str_data):
    carry = ''.join(str_data.split(','))
    return carry

print(convert_int("1,234,567"))
print()

# Q.236
def 함수(num):
    return num +4

a = 함수(10)
b = 함수(a)
c = 함수(b)
print(c)
# 22
print()

# Q.237
def 함수(num):
    return num + 4

c = 함수(함수(함수(10)))
print(c)
# 22
print()

# Q.238
def 함수1(num):
    return num + 4

def 함수2(num):
    return num * 10

a = 함수1(10)
c = 함수2(a)
print(c)
# 140
print()

# Q.239
def 함수1(num):
    return num + 4

def 함수2(num):
    num = num + 2
    return 함수1(num)

c = 함수2(10)
print(c)
# 16
print()

# Q.240
def 함수0(num):
    return num * 2

def 함수1(num):
    return 함수0(num + 2)

def 함수2(num):
    num = num + 10
    return 함수1(num)

c = 함수2(2)
print(c)
# 28
print()

