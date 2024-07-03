# Q.151
리스트 = [3, -20, -3, 44]
for i in 리스트:
    if (i < 0):
        print(i)
print()

# Q.152
리스트 = [3, 100, 23, 44]
for i in 리스트:
    if ((i%3) == 0):
        print(i)
print()

# Q.153
리스트 = [13, 21, 12, 14, 30, 18]
for i in 리스트:
    if (((i%3)==0) and (i<20)):
        print(i)
print()

# Q.154
리스트 = ["I", "study", "python", "language", "!"]
for i in 리스트:
    if (3 <= len(i)):
        print(i)
print()

# Q.155
리스트 = ["A", "b", "c", "D"]
for i in 리스트:
    if ('A' <= i <= 'Z'):
        print(i)
print()
# if (i.isupper()): print(i)

# Q.156
for i in 리스트:
    if('a' <= i <= 'z'):
        print(i)
print()
# if not (i.isupper()): print(i)

# Q.157
리스트 = ['dog', 'cat', 'parrot']
for i in 리스트:
    c = i[0].upper()
    print(c + i[1:])
print()

# Q.158
리스트 = ['hello.py', 'ex01.py', 'intro.hwp']
for i in 리스트:
    d = i.split('.')
    print(d[0])
print()

# Q.159
리스트 = ['intra.h', 'intra.c', 'define.h', 'run.py']
for i in 리스트:
    d = i.split('.')
    if (d[1] == 'h'):
        print(i)
print()

# Q.160
for i in 리스트:
    d = i.split('.')
    if ((d[1] == 'h') or (d[1] == 'c')):
        print(i)
print()

