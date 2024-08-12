# 정규 표현식 예제

# 정규 표현식 객체 사용
# complie(pattern):
#  - 동일 패턴을 여러 번 검색하는 경우, 편리하게 사용
#  - re 모듈 함수들은 pattern 파라미터 없이 호출가능
#   예) search(string, pos), match(string, pos)

import re
print()

# complie() 사용 안함
m = re.match('[a-z]+', 'Python')
print(m)
print(re.search('apple', 'I like apple!'))
print(re.match('[a-z]+', 'pythoN'))
print()

# complie() 사용: 객체 생성
p = re.compile('[a-z]+')    # 알파벳 소문자
m = p.match('python')
print(m)
print(p.search('I like apple 123'))
print()

# match(): 문자열의 처음부터 검사, 소문자를 인식
m = re.match('[a-z]+', 'pythoN')    # 시작이 소문자
print(m)
m = re.match('[a-z]+', 'PYthon')    # 시작이 대문자
print(m)
print()

print(re.match('[a-z]+', 'regex python'))
print(re.match('[a-z]+', 'regexpython'))
print(re.match('[a-z]+', 'regexpythoN'))
print(re.match('[a-z]+$', 'regexpythoN'))   # 문자열 마지막을 검사?
print(re.match('[a-z]+', 'regexPython'))
print(re.match('[a-z]+$', 'regexPython'))
print()

# findall(): 일치하는 모든 문자열을 리스트로 리턴
p = re.compile('[a-z]+')
print(p.findall('life is too short! Regula expression test'))
print()

# search(): 일치하는 첫 번째 문자열만 리턴
result = p.search('I like apple 123')
print(result)

result = p.findall('I like apple 123')
print(result)
print()

# match(): 객체 메소드 -> group

# 전화번호 분석
# - 전화번호: '지역번호-국번-전화번호'
# - groups(): 매칭되는 문자열의 전체 그룹을 리턴

# ^ ...$을 명시해야 정확한 자리수 검사가 이루어짐
tel_checker = re.compile(r'^(\d{2,3})-(\d{3,4})-(\d{4})$')

print(tel_checker.match('02-123-4567'))
print()

match_groups = tel_checker.match('02-123-4567').groups()
match_group = tel_checker.match('02-123-4567').group()
print(match_group)
print(match_groups)
print()

m = tel_checker.match('02-123-4567')
print(m.groups())
print('group(): ', m.group())
print('group(0): ', m.group(0))
print('group(1): ', m.group(1))
print('group(2, 3): ', m.group(2, 3))
print('start(): ', m.start())   # 매칭된 문자열의 시작 인덱스
print('end(): ', m.end())   # 매칭된 문자열의 마지막 인덱스
print()

print(tel_checker.match('053-950-45678'))
print(tel_checker.match('053950-4567'))
print()

# 전화번호에서 dash(-) 제거하고 검사하기
tel_number = '053-950-4567'
tel_number = tel_number.replace('-', '')
print(tel_number)

tel_checker1 = re.compile(r'^(\d{2,3})(\d{3,4})(\d{4})$')
print(tel_checker1.match(tel_number))
print(tel_checker1.match('0239501234'))
print()

# 휴대 전화번호 매칭
# 휴대 전화번호 구성: 사업자번호(3자리)-국번(3,4자리)-전화번호(4자리)
# - (?:0|1|[6-9]): 뒤에 따라 나오는 숫자(0|1|6|7|8|9)를 하나의 그룹으로 합침
call_phone = re.compile(r'^(01(?:0|1|[6-9]))-(\d{3,4})-(\d{4})$')

print(call_phone.match('010-123-4567'))
print(call_phone.match('010-1234-5678'))
print(call_phone.match('010-123-4567'))
print(call_phone.match('010-12345678'))
print()

# 전방 탐색(lookahead)
# 전방 긍정 탐색: 패턴과 일치하는 문자열을 만나면 패턴 앞의 문자열 반환: (?=패턴)
# 전방 부정 탐색(?!): 패턴과 일치하지 않는 문자열을 만나면 패턴 앞의 문자열 반환: (?!패턴)

# 전방 긍정 탐색: (문자열이 won을 포함하고 있으면 won 앞의 문자열을 반환)
lookahead1 = re.search('.+(?=won)', '1000 won')
if lookahead1 != None:
    print(lookahead1.group())
else:
    print('None')
lookahead2 = re.search('.+(?=am)', '2023-01-26 am 10:00:01')
print(lookahead2)
# 전방 부정 탐색(?!): 4자리 숫자 다음에 '-'를 포함하지 않으면 앞의 문자열 리턴
lookahead3 = re.search('\d{4}(?!-)', '010-1234-5678')
print(lookahead3)
print()

# 후방 탐색(lookbehind)
# 후방(?<) 긍정(=) 탐색: 패턴과 일치하는 문자열을 만나면 패턴 뒤의 문자열 반환: (?<=패턴)
# 후방(?<) 부정(!) 탐색: 패턴과 일치하지 않는 문자열을 만나면 패턴 뒤의 문자열 반환: (?<!패턴)

# 후방 긍정 탐색 ('am' 다음에 문자가 1개 이상 있으면 해당 문자열을 리턴)
lookbehind1 = re.search('(?<=am).+', '2023-01-26 am 11:10:01')
print(lookbehind1)

lookbehind2 = re.search('(?<=:).+', 'USD:$51')
print(lookbehind2)

# 후방 부정 탐색('\b': 공백)
# 공백 다음에 $ 기호가 없고 숫자가 1개 이상이고 공백이 있는 경우
lookbehind3 = re.search(r'\b(?<!\$)\d+\b', 'I paid $30 fro 100 apples.')
print(lookahead3)
print()


