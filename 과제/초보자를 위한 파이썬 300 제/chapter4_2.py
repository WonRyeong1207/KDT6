# Q.61
price = ['20180728', 100, 130, 140, 150, 160, 170]
print(price[1:])
print('\n')

# Q.62
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(nums[::2])
print('\n')

# Q.63
print(nums[1::2])
print('\n')

# Q.64
nums = [1, 2, 3, 4, 5]
print(nums[::-1])
print('\n')

# Q.65
interest = ['삼성전자', 'LG전자', 'Naver']
print(interest[::2])
print(interest[0], interest[2])
print('\n')

# Q.66
interest.extend(['SK하이닉스', '미래에셋대우'])
print(" ".join(interest))
print('\n')

# Q.67
print("/".join(interest))
print('\n')

# Q.68
print('\n'.join(interest))
print('\n')

# Q.69
string = "삼성전자/LG전자/Naver"
interest = list(string.split('/'))
print(interest)
print('\n')

# Q.70
data = [2, 4, 3, 1, 5, 10, 9]
print(sorted(data))
