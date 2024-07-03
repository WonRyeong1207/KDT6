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

with open('EX_PY06/words.txt', 'r') as words_file:
    word = words_file.readlines()
    
    for i, d in enumerate(word):
        word[i] = d.rstrip('\n')
    
    for i in word:
        if (i == i[::-1]):
            print(i)

