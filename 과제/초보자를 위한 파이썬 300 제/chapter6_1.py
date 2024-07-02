# Q.81
scores = [8.8, 8.9, 8.7, 9.2, 9.7, 9.9, 9.5, 7.8, 9.4]
*valid_score, _, _ = scores
print(valid_score)
print()

# Q.82
_, _, *valid_score = scores
print(valid_score)
print()

# Q.83
_, *valid_score, _ = scores
print(valid_score)
print()

# Q.84
temp = {}
print(temp)
print()

# Q.85
icecream = {'메로나':1000, '폴라포':1200, '빵빠레':1800}
print(icecream)
print()

# Q.86
icecream['죠스바'] = 1200; icecream['월드콘'] = 1500
print(icecream)
print()

# Q.87
ice = {'메로나':1000,
       '폴로포':1200,
       '빵빠레':1800,
       '죠스바':1200,
       '월드콘':1500}

print(f"메로나 가격 : {ice['메로나']}")
print()

# Q.88
ice['메로나'] = 1300
print(ice)
print()

# Q.89
del ice['메로나']
print(ice)
print()

# Q.90
# 없는 키값을 불렀기 때문에 오류가 났음.
# 또는 값을 지정해주지 않았기 때문에
icecream['누가바'] = 1000
print(icecream)
print()

