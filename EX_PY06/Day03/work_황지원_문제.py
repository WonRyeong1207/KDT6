# Q.1
print("Hello World")
print('\n')

# Q.2
print("Mary's cosmetics")
print('\n')

# Q.3
print('신씨가 소리를 질렀다. "도둑이야".')
print('\n')

# Q.4
print('C:\\Windows')
print('\n')

# Q.5
print("안녕하세요.\n만나서\t\t반갑습니다.")
# \n은 줄바꿈이고 \t는 탭공백을 만든다.
print('\n')

# Q.6
print("오늘은", "일요일")
# 오늘은 일요일
print('\n')

# Q.7
print("naver", "kakao", "sk", "samsung", sep=';')
print('\n')

# Q.8
print("naver", "kakao", "sk", "samsung", sep='/')
print('\n')
# Q.9
print("first", end=''); print("Second")
print('\n')

# Q.10
print(5/3)
print('\n')

# Q.11
삼성전자 = 50000
주식 = 10
총평가금액 = 삼성전자 * 주식
print(f"총 평가금액 : {총평가금액}")
print('\n')

# Q.12
시가총액 = 2980000000000000
현재가 = 50000
PER = 15.79

print(시가총액, type(시가총액))
print(현재가, type(현재가))
print(PER, type(PER))
print('\n')

# Q.13
s = 'hello'
t = 'python'
print(f"{s}! {t}")
print('\n')

# Q.14
# 2 + 2 * 3 = 8일 것이다
print(2 + 2 * 3)
print('\n')

# Q.15
a = 128
print(type(a))
a = "132" # type <class 'str'>
print(type(a))
print('\n')

# Q.16
num_str = '720'
num_str = int(num_str)
print(num_str, type(num_str))
print('\n')

# Q.17
num = 100
num = str(num)
print(num, type(num))
print('\n')

# Q.18
n = '15.79'
n = float(n)
print(n, type(n))
print('\n')

# Q.19
year = '2020'
year = int(year)
print(year-1, year-2, year-3)
print('\n')

# Q.20
month = 36
price = 48584
total_price = month * price
print(total_price)
print('\n')

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
print('\n')

# Q.31
a = '3'
b = '4'
print(a + b)
# 34
print('\n')

# Q.32
print("Hi" * 3)
# HiHiHI
print('\n')

# Q.33
print('-' * 80)
print('\n')

# Q.34
t1 = 'python'
t2 = 'java'
print((t1 + t2) * 4)
print('\n')

# Q.35
name1 = '김민수'
age1 = 10
name2 = '이철희'
age2 = 13
print("이름: %s 나이: %d" % (name1, age1))
print("이름: %s 나이: %d" % (name2, age2))
print('\n')

# Q.36
print("이름: {} 나이: {}".format(name1, age1))
print("이름: {} 나이: {}".format(name2, age2))
print('\n')

# Q.37
print(f"이름: {name1} 나이: {age1}")
print(f"이름: {name2} 나이: {age2}")
print('\n')

# Q.38
상장주식수 = '5,969,782,550'
컴마제거 = 상장주식수.replace(',', "")
타입변환 = int(컴마제거)
print(타입변환, type(타입변환))
print('\n')

# Q.39
분기 = '2020/03(E) (IFRS연결)'
print(분기[:7])
print('\n')

# Q.40
data = '    삼성전자    '
data = data.strip()
print(data)
print('\n')

# Q.41
ticker = 'btc_krw'
ticker = ticker.upper()
print(ticker)
print('\n')

# Q.42
ticker = 'BTC_KRW'
ticker = ticker.lower()
print(ticker)
print('\n')

# Q.43
a = 'hello'
a = a.capitalize()
print(a)
print('\n')

# Q.44
file_name = '보고서.xlsx'
file_name.endswith('xlsx')
print(file_name.endswith('xlsx'))
print('\n')

# Q.45
file_name.endswith(('xlsx', 'xls'))
print(file_name.endswith(('xlsx', 'xls')))
print('\n')

# Q.46
file_name = '2020_보고서_xlsx'
file_name.startswith('2020')
print(file_name.startswith('2020'))
print('\n')

# Q.47
a = 'Hello world'
a, b = a.split()
print(a, b)
print('\n')

# Q.48
ticker = 'btc_krw'
c, d = ticker.split('_')
print(c, d)
print('\n')

# Q.49
data = '2020-05-01'
year, month, day = data.split('-')
print(year, month, day)
print('\n')

# Q.50
data = '039490    '
data = data.rstrip()
print(data * 2)
print('\n')

# Q.51
movie_rank = ['닥터 스트레인지', '스플릿', '럭키']
print(movie_rank)
print('\n')

# Q.52
movie_rank.append('배트맨')
print(movie_rank)
print('\n')

# Q.53
movie_rank.insert(1, '슈퍼맨')
print(movie_rank)
print('\n')

# Q.54
del movie_rank[3]
print(movie_rank)
print('\n')

# Q.55
del movie_rank[2:]
print(movie_rank)
print('\n')

# Q.56
lang1 = ['C', 'C++', 'JAVa']
lang2 = ['Python', 'Go', 'C#']
langs = lang1 + lang2
print(langs)
print('\n')

# Q.57
nums = [1, 2, 3, 4, 5, 6, 7]
print(f"max: {max(nums)}")
print(f"min: {min(nums)}")
print('\n')

# Q.58
nums = [1, 2, 3, 4, 5]
print(sum(nums))
print('\n')

# Q.59
cook = ['피자', '김밥', '만두', '양념치킨', '족발', '피자', '김치만두', '쫄면', '소시지', '라면', '팥빙수', '김치전']
print(len(cook))
print('\n')

# Q.60
print(sum(nums)/len(nums))
print('\n')

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
print('\n')

# Q.71
my_variable = []
print(my_variable)
print('\n')

# Q.72
movie_rank = ('닥터 스트레인지', '스플릿', '럭키')
print(movie_rank)
print('\n')

# Q.73
t = (1,)
print(t)
print('\n')

# Q.74
t = (1, 2, 3)
# t[0] = 'a'
# 튜플은 리스트처럼 요소를 변경할 수 있는 자료형이 아니기 때문에 요소를 변경 삭제 할 수 없음.

# Q.75
t = 1, 2, 3, 4
# t의 타입은 튜플이다. 파이썬은 저렇게 값을 입력하면 자동으로 튜플로 만듦.
print(f"t의 자료형: {type(t)}")
print('\n')

# q.76
t = ('a', 'b', 'c')
# 값을 변경 할 수 없기에 재 지정 t = ('A', 'B', 'C')
t = str(''.join(t))
t = t.upper()
t = ','.join(t)
t = tuple(t.split(','))
print(t) # 이렇게 혈식을 변경해서 저장하면 가능
print('\n')

# Q.77
interest = ('삼성전자', 'LG전자', 'SK Hynix')
interest = list(interest)
print(f"interest의 자료형: {type(interest)}")
print('\n')

# Q.78
interest = tuple(interest)
print(f"interest의 자료형: {type(interest)}")
print('\n')

# Q.79
temp = ('apple', 'banana', 'cake')
a, b, c = temp
print(a, b, c)
# 오류가 난다?
# 오류 안아도 split되어 저장됨.
print('\n')

# Q.80
t = tuple(range(2, 99, 2))
print(t)
