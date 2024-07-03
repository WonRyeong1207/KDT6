# for문 하나로 구구단 출력

for dan in range(1, 101):
    if (dan//10 == 2):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 3):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 4):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 5):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 6):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 7):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 8):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
    elif (dan//10 == 9):
        if not (dan%10 == 0):
            print(f"{dan//10} * {dan%10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"\n- {dan//10} 단 -", end='\t')
print('\n')

# 구구단 전체 출력
# 굳이 for문 하나만 쓸필요가 없었어...
for dan in range(1, 101):
    if (dan%10 == 2):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 3):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 4):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 5):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 6):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 7):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 8):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    elif (dan%10 == 9):
        if not (dan//10 == 0):
            print(f"{dan%10} * {dan//10} = {(dan%10)*(dan//10):2}", end='\t')
        else: print(f"-- {dan%10} 단 --", end='\t')
    else:
        if (dan%10==0):
            print()
print()

# 이중 for문을 사용 한다면
for i in range(10):
    for d in range(2, 10):
        if (i == 0):
            print(f"-- {d} 단 --", end='\t')
        else:
            print(f"{d} * {i} = {d*i:2}", end='\t')
    print()
