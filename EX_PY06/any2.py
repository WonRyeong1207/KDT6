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

for dan in range(1, 101):
    
    if (dan//10 == 2):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 3):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 4):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 5):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 6):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 7):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 8):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
    if (dan//10 == 9):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")


print('\n\n\n')


for dan in range(1, 101):
    
    if (dan%10 == 2):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 3):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 4):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 5):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 6):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 7):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 8):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    elif (dan%10 == 9):
        print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='    ')
    else:
        print()
        
