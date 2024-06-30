# Q.21
letters = 'python'
print(letters[0], letters[2])
print('\n')

# Q.22
license_plate = "24가 2210"
print(license_plate[4:])
print('\n')

# Q.23
string = '홀짝홀짝홀짝'
print(string[::2])
print('\n')

# Q.24
string = 'PYTHON'
print(string[::-1])
print('\n')

# Q.25
phone_number = '010-1111-2222'
phone_number = phone_number.replace('-', ' ')
print(phone_number)
print('\n')

# Q.26
phone_number = phone_number.replace(' ', '')
print(phone_number)
print('\n')

# Q.27
url = 'http://sharebook.kr'
url = url.split('.')
print(url[-1])
print('\n')

# Q.28
lang = 'python'
# lang[0] = 'P'
print(lang)
# TyprError
print('\n')

# Q.29
string = 'abcde2a354a32a'
string = string.replace('a', 'A')
print(string)
print('\n')

# Q.30
string = 'abcd'
string.replace('b', 'B')
print(string)
# aBcd 아니었음 abcd