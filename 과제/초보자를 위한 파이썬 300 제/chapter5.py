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