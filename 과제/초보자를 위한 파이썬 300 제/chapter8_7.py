# Q. 191
data = [
    [2000, 3050, 2050, 1980],
    [7500, 2050, 2050, 1980],
    [15400, 15050, 1550, 14900]
]
for row in data:
    for col in row:
        print(col * 1.00014)
print()

# Q.192
for row in data:
    for col in row:
        print(col * 1.00014)
    print('-' * 4)
print()

# Q.193
result = []
for row in data:
    for col in row:
        result.append(col * 1.00014)
print(result)
print()

# Q.194
result = []
for row in range(len(data)):
    sub = []
    for col in range(len(data[row])):
        sub.append(data[row][col] * 1.00014)
    result.append(sub)
print(result)
print()

# Q.195
ohlc = [["open", "high", "low", "close"],
        [100, 110, 70, 100],
        [200, 210, 180, 190], 
        [300, 310, 300, 310]]
for i in range(1, len(ohlc)):
    print(ohlc[i][-1])
print()

# Q.196
for i in range(1, len(ohlc)):
    if (150 < ohlc[i][-1]):
        print(ohlc[i][-1])
print()

# Q.197
for i in range(1, len(ohlc)):
    if (ohlc[i][0] <= ohlc[i][-1]):
        print(ohlc[i][-1])
print()

# Q.198
volatility = []
for i in range(1, len(ohlc)):
    volatility.append(ohlc[i][1] - ohlc[i][2])
print(volatility)
print()

# Q.199
volatility = 0
for i in range(1, len(ohlc)):
    if (ohlc[i][0] < ohlc[i][-1]):
        volatility = ohlc[i][1] - ohlc[i][2]
        print(volatility)
print()

# Q.200
volatility = []
for i in range(1, len(ohlc)):
    volatility.append(ohlc[i][0] - ohlc[i][-1])
print(sum(volatility))
print()

