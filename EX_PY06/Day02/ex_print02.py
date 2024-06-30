# 내장함수 print() 사용법2
# - print( parameter )
# - print 함수의 매개변수 알아보기
# - 매개변수(parameter) : 함수 코드 실행 시에 필요한 데이터를 명시해 놓은 것
# - sep : 구분자, 여러개의 데이터를 보기 좋게 출력되도록 구분해주는 변수
# - end : 출력 데이터의 마지막에 줄바꿈 문자를 추가해 놓은 변수, 줄바꿈보다는 줄안바꿈 같은디...

year, month, day, hour, minute, second = '2024', '06', '27', '10', '56', '53'

print(year, month, day, sep='-', end='T')
print(hour, minute, second, sep=':')

# 여러개의 데이터 전달 시 구분 문자 넣기
f_num, m_num, l_num = '010','1234', '1234'
print(f_num, m_num, l_num, sep='-')

# 화면 풀력 후에 문자 설정하기 => 기본 줄바꿈 '\n'
print(1); print(2); print(3)
# 기본으로 \n이 되어있는 형태
print(1); print(2, end=' '); print(3) # 2번째 이후에는 공백뒤에 출력을 이어감. 즉, \n을 공백으로 치환함.

# 출력결과는 다음과 같고 print 4개 사용
# 1234567
# abcdefg ABCDEFG
# 123456
print(1234567)
print('abcdefg', end=' ')
print('ABCDEFG')
print(1234567)
