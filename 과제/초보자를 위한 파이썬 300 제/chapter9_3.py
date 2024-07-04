# q.221
def print_reverse(data):
    print(data[::-1])
print_reverse("I want to meet karesi")
print()

# Q.222
def print_score(data):
    print(sum(data)/len(data))
print_score([10, 20, 30])
print()

# Q.223
def print_even(data):
    for i in data:
        if (i%2 == 0):
            print(i)
print_even([1, 3, 2, 10, 12, 11, 15])
print()

# Q.224
def print_keys(data):
    for i in data.keys():
        print(i)
print_keys({"이름":"김말똥", "나이":30, "성별":0})
print()

# Q.225
my_dic = {"10/26" : [100, 130, 100, 100],
          "10/27" : [10, 12, 10, 11]}

def print_value_by_key(dic_data, key):
    print(dic_data[key])
print_value_by_key(my_dic, "10/26")
print()

# Q.226
def print_5xn(data):
    cnt = 0
    carry = ''
    for i in range(len(data)):
        if cnt == 5:
            carry = carry + '\n' + data[i]
            cnt = cnt + 1
        else:
            carry = carry + data[i]
            cnt = cnt + 1
    print(carry)
    
print_5xn("아이엠어보이유알어걸")
print()

# Q.227
def print_mxn(data, num):
    cnt = 0
    carry = ''
    for i in range(len(data)):
        if cnt == num:
            carry = carry + '\n' + data[i]
            cnt += 1
        else:
            carry += data[i]
            cnt += 1
    print(carry)
    
print_mxn("아이엠어보이유알어걸", 3)
print()

