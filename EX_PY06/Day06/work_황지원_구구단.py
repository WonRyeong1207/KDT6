# for문 하나로 구구단 출력

for dan in range(1, 101):
    if (dan//10 == 2):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 3):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 4):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 5):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 6):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 7):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 8):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
    elif (dan//10 == 9):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10)}")
        else: print(f"\n-- {dan//10} 단--")
print()

# 구구단 전체 출력
# 굳이 for문 하나만 쓸필요가 없었어...
for dan in range(1, 101):
    if (dan%10 == 2):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 3):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 4):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 5):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 6):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 7):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 8):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    elif (dan%10 == 9):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10)}", end='\t')
        else: print(f"-- {dan%10} 단--", end='\t')
    else:
        if (dan%10==0):
            print()
print()

'''
for dan in range(2, 10):
    for i in range(10):
        if (i == 0):
            print("--- {dan} ---", end='\t')
        else:
            print(f"{dan} * {i} = {dan*i}|, end='\t')
    print()
'''