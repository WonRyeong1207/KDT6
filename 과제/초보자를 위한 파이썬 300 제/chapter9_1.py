# Q. 201
def print_coin():
    print("비트코인")

# Q.202
print_coin()
print()

# Q.203
for i in range(100):
    print_coin()
print()

# Q.204
def print_coins():
    for i in range(100):
        print("비트코인")
        
# Q.205
# hello()
def hello():
    print("Hi")
hello()
# 함수 선언전에 불렀기 때문에 인식하지 못한것임.
print()

# Q.206
def message():
    print("A")
    print("B")
    
message()
print("C")
message()
# A\nB\nC\nA\nB
print()

# Q.207
print("A")
def message():
    print("B")
print("C")
message()
# A\nC\nB
print()

# 208
print("A")
def message1():
    print("B")
print("C")
def message2():
    print("D")
message1()
print("E")
message2()
#A\nC\nB\nE\nD
print()

# Q.209
def message1():
    print("A")
    
def message2():
    print("B")
    message1()

message2()
# B\nA
print()

# Q.210
def message1():
    print("A")

def message2():
    print("B")

def message3():
    for i in range(3):
        message2()
        print("C")
    message1()
    
message3()
# B\nC\nB\nC\nB\nC\nA
print()

