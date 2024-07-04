# 22장, 25장

# p.270 연습문제
a = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india']
b = [x for x in a if (len(x) == 5)]
print(b)
print()

# p.271 심사문제
num1, num2 = map(int, input().split())
# num1, num2 = 10, 20

# sq_list = [2**x for x in range(num1, num2+1)]
sq_list = [pow(2, x) for x in range(num1, num2+1)]

sq_list.pop(1)
sq_list.pop(-2)

# del sq_list[1]
# del sq_list[-2]

print(sq_list)
print()

# p.328 연습문제
maria = {'koearn':94, 'english':91, 'mathematics':89, 'science':83}

average = sum(maria.values()) / len(maria)
print(average)
print()

# p.328~329 심사문제
keys = input().split()
values = input().split()
# keys = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf']
# values = [30, 40, 50, 60, 70, 80, 90]

x = dict(zip(keys, values))

x = {key:value for key, value in x.items() if not ((key=='delta') or (value==30))}
print(x)
print()
