# 형변환 / 타입캐스팅
# - 자료형을 다른 종류의 자료형으로 변경
# - 종류
#   자동 / 묵시적 형변환 : 컴퓨터가 진행
#   수동 / 명시적 형변환 : 개발자가 진행

age = 23.6

# float ---> int
print(age, int(age), type(age))
age = int(age)
print(age, type(age))

# int ---> float
age = float(age)
print(age, type(age))

# float ---> str
age = str(age)
print(age, type(age))

'''
key_list = ['health', 'mana']
value_list = [573.6, 308.8]
hall = dict(zip(key_list, value_list))
'''
# 넓게 주석한다고 배웠는데 긴 문자열이었음 ㅋㅋㅋㅋㅋ