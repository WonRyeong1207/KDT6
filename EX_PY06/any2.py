'''
keys = ['alpha', 'bravo', 'charlie', 'delta']
values = [10, 20, 30, 40]

x = dict(zip(keys, values))
x = {key:value for key, value in x.items() if not ((key == 'delta') or (value == 30))}
print(x)
'''


# 중첩 for문을 하나로

'''
dan = [[x * i for x in range(1,10)] for i in range(2, 10)]
print(dan)
print('\n\n')

n2, n3, n4, n5, n6, n7, n8, n9 = enumerate(dan)
print(n2)
'''
'''
with open('EX_PY06/words.txt', 'r') as word_file:
    words_list = list(map(str, word_file.read().split()))
    # print(words_list)
    for i, d in enumerate(words_list):
        words_list[i] = d.rstrip(',.')
        print(words_list[i], end=' ')
    print('n')
    
    for d in words_list:
        if ('c' in d):
            print(d)
'''
'''
with open('EX_PY06/words.txt', 'r') as words_file:
    word = words_file.readlines()
    
    for i, d in enumerate(word):
        word[i] = d.rstrip('\n')
    
    for i in word:
        if (i == i[::-1]):
            print(i)
'''
'''
k, e, m ,s = 76, 82, 89, 84
def get_average(korean=0, english=0, mathematics=0, science=0):
    cnt = 0
    if korean:
        cnt += 1
    if english:
        cnt += 1
    if mathematics:
        cnt += 1
    if science:
        cnt += 1
    return (korean + english + mathematics + science)/cnt
average = get_average(korean=k, english=e, mathematics=m, science=s)
print(average)
average = get_average(english=e, science=s)
print(average)
'''
'''
def fib(num):
    if (num == 0):
        return 0
    elif (num == 1):
        return 1
    else:
        return fib(num-2) + fib(num-1)
    
n = fib(10)
print(n)

num = 10
total = [0, 1]
for i in range(2, num+1):
    total.append(total[i-2] + total[i-1])
print(total[-1])
'''
'''
n = 5
for i in range(n):
    for j in range(n):
        if (i < j):
            print(' ', end='')      
    print("*" * (1 + (i*2)), end='')
    print()
print()
'''

files = ['97.xlsx', '98.docx', '99.docx', '100.docx', '101.docx', '102.docx', '1.jpg', '10.png', '11.png', '2.jpg', '3.png']
k = list(map(lambda x: x.zfill((3-x.find('.')+len(x))), files))
print(k)
print()

def counntdown(n):
    i = n + 1
    def count():
        nonlocal i
        i -= 1
        return i
    return count

n = 20
c = counntdown(n)
for i in range(n):
    print(c(), end=' ')
print()

