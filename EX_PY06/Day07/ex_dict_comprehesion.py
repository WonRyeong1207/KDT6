# dict의 내포

keys = ['a', 'b', 'c', 'd']

x = {key:value for key, value in dict.fromkeys(keys).items()}
print(x)
