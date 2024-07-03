# Q.171
price_list = [32100, 32150, 32000, 32500]
for i in range(len(price_list)):
    print(price_list[i])
print()

# Q.172
for i in range(len(price_list)):
    print(i, price_list[i])
print()

# Q.173
for i in range(len(price_list)):
    print(3-i, price_list[3-i])
print()

# Q.174
for i in range(1, len(price_list)):
    print(90+(10*i), price_list[i])
print()

# Q.175
my_list = ["가", "나", "다", "라"]
for i in range(len(my_list)-1):
    print(my_list[i], my_list[i+1])
print()

# Q.176
my_list = ["가", "나", "다", "라", "마"]
for i in range(len(my_list)-2):
    print(my_list[i], my_list[i+1], my_list[i+2])
print()

# Q.177
my_list = ["가", "나", "다", "라"]
c = my_list[::-1]
for i in range(len(my_list)-1):
    print(c[i], c[i+1])
print()

# Q.178
my_list = [100, 200, 400, 800]
for i in range(len(my_list)-1):
    print(my_list[i+1] - my_list[i])
print()

# Q.179
my_list = [100, 200, 400, 800, 1000, 1300]
for i in range(len(my_list)-2):
    print((my_list[i] + my_list[i+1] + my_list[i+2])/3)
print()

# Q.180
low_prices = [100, 200, 400, 800, 1000]
high_prices = [150, 300, 430, 880, 1000]
volatility = []

for i in range(5):
    volatility.append(high_prices[i]-low_prices[i])
print(volatility)
print()

