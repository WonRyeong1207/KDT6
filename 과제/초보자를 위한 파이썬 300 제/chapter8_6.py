# Q.181
apart = [['101호', '102호'], ['201호', '202호'], ['301호', '302호']]
print(apart)
print()

# Q.182
stock = [['시가', 100, 200, 300], ['종가', 80, 210, 330]]
print(stock)
print()

# Q.183
stock = {'시가':[100, 200, 300],
         '종가':[80, 210, 300]}
print(stock)
print()

# Q.184
stock = {'10/10':[80, 110, 70, 90], '10/11':[210, 230, 190, 200]}
print(stock)
print()

# Q.185
apart = [[101, 102], [201, 202], [301, 302]]
for row in apart:
    for col in row:
        print(col, "호")
print()

# Q.186
for row in apart[::-1]:
    for col in row:
        print(col, "호")
print()

# Q.187
for row in apart[::-1]:
    for col in row[::-1]:
        print(col, "호")
print()

# Q.188
for row in apart:
    for col in row:
        print(col, "호")
        print('-'*5)
print()

# Q.189
for row in apart:
    for col in row:
        print(col, "호")
    print('-'*5)
print()

# Q.190
for row in apart:
    for col in row:
        print(col, "호")
print('-'*5)
print()

