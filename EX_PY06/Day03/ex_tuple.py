# Tuple 사용법
# - 다양한 종류의 여러 개 데이터를 저장하는 타입
# - 리스트와 비슷하지만 수정, 삭제 안됨!!!
#  - 형식 : (1, 2, ...) 한개일때 (1,) 또는 1,

# 튜플 데이터 생성
datas =()
print(type(datas), datas, len(datas))

datas = (1, 5, 7)
print(type(datas), datas, len(datas))

datas = (1,)
print(type(datas), datas, len(datas))

datas = 1,
print(type(datas), datas, len(datas))

# 튜플 데이터의 원소/요소 읽기
datas = 11,22,33,44,55 # index가 존재

print(f"datas[2] : {datas[2]}")

# 원소/요소 수정 및 삭제 즉, 변경불가!!!
# datas[-1] = 'a'

# 튜플 데이터의 원소/요소 변경 ==> 형변환
birthday = (1999, 1, 1)
print(birthday, type(birthday))
# 1월을 7월로 바꾸고 싶음
birthday = list(birthday)
birthday[1] = 7
print(birthday, type(birthday))
birthday = tuple(birthday)
print(birthday, type(birthday))
