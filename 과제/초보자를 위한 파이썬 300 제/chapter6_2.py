# Q.91
inventory = {'메로나':[300, 20],
             '비비빅':[400, 3],
             '죠스바':[250, 100]}
print(inventory)
print()

# Q.92
print(f"{inventory['메로나'][0]} 원")
print()

# Q.93
print(f"{inventory['메로나'][1]} 개")
print()

# Q.94
inventory['월드콘'] = [500, 7]
print(inventory)
print()

# Q.95
icecream = {'탱크보이':1200, '폴라포':1200, '빵빠레':1800, '월드컵':1500, '메로나':1000}
key_list = list(icecream.keys())
print(key_list)
print()

# Q.96
value_list = list(icecream.values())
print(value_list)
print()

# Q.97
print(sum(value_list))
print()

# Q.98
new_product = {'팥빙수':2700, '아맛나':1000}
icecream.update(new_product)
print(icecream)
print()

# Q.99
keys = ('apple', 'pear', 'peach')
vals = (300, 250, 400)

result = dict(zip(keys, vals))
print(result)
print()

# Q.100
date = ['09/05', '09/06', '09/07', '09/08', '09/09']
close_price = [10500, 10300, 10100, 10800, 11000]

close_table = dict(zip(date, close_price))
print(close_table)
print()

