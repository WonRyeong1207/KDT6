# Collection 자료형에 공통적인 부분 살펴보기

# 여러개의 변수에 데이터 저장
# name = '홍길동'
# age = 12
# job = '의적'
# gender = '남'

data = '홍길동', 12, '의적', '남'

# 변1, 변2, 변3, ... ==?> 언 팩킹
name, age, job, gender = '홍길동', 12, '의적', '남'
# 변수명 개수와 데이터 수는 동일해야함.

name, age, _, _ = '괴도키드', 17, '도둑', '남' # 데이터는 있는데 변수가 필요하지 않을때 의미가 없을때

print(name, age, _)

score = [100, 99]
korean, math = [100, 99]
print(score, korean, math) # 개수만 맞으면 리스트도 가능

person = {'name':'박', 'age':11}
k1, k2 = {'name':'박', 'age':11}
print(person, k1, k2)

print('\n\n')
# 생성자 : 타입명과 함수명이 동일
# - int(), float(), str(), bool(), list(), set(), map(), tuple(), range()

# 기본 데이터 타입
num = int(10)
fnum = float(10.2)
msg = str('Good')
isOk = bool(False) # 이렇게 안 적어도 알아서 형변환해서 저장해줌
print(num, fnum, msg, isOk)

# 컬렉션 데이터 타입
lunms = list([1, 2, 3])
tnums = tuple((3, 6, 9))
ds = dict({'d1':10, 'd2':20})
ss = set({1, 2, 1, 2, 3, 4, 5, 8})
print(lunms, tnums, ds, ss)

# 타입변경 => 형변환
# dict 자료형은 다른 자료값과 달리 데이터 형태 다름
# - 데이터 형태 => 키:값
# ds = dict([1, 2, 3]) 에러남
ds = dict(n1=1, n2=2, n3=3)
# 이런 형식으로 지정할때는 키에 ''를 주면 에러, 숫자로 키를 지정해도 에러, str만 가능
print(ds)

ds = dict([('name','nana'), ('age', 13)]) # 키와 값이 같이 한 리스트에 있다면 딕셔너리로 저장
print(ds)

# 내장함수 zip() : 같은 인덱스의 데이터까리 묶음
key_list = ['name', 'age', 'gender']
value_list = ['kiki', 13, 'female']
print(list(zip(key_list, value_list)))

