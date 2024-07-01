# set 자료형
# - 여러 가지 종류의 여러 개 데이터를 저장
# - 단! 중복 안됨!!!
# - 컬렉션 타입의 데이터 저장 시 Tuple 가능
# - 형태 : {데이터1, 데이터2, ...}

# set 생성
data = []
data = ()
data = {}
data = set()
print(f"data의 타입 : {type(data)}, 원소의 개수 : {len(data)}, 데이터 : {data}")

data = [10, 20, 30, -10, 20, 30, 40]
print(set(data))
# 중복 데이터 였던거 같음. 중복을 제거할 수 있음.

data = {9.34, 'Apple', 10, True, '10'}
print(f"data : {set(data)}")

data1 = {1, 2, 3, (1, 2)}
data2 = {1, 2, 3, (1)}
print(f"data1의 타입: {type(data1)}, {set(data1)}")
print(f"data2의 타입: {type(data2)}, {set(data2)}")

# set 내장함수
data = {1, 2, 3} # ===> set([1, 2, 3])
data = set({1, 2})
data = set([1, 2, 2, 4])
data = set("Good")
print(data)

data = set({'name':'nana', 'age':12}) # key 중복이 되면 하나는 버려짐. 뒤에 들어온 key 값을 사용함.
print(data)

# set은 중복된 값은 안 가져감.
# set의 데이터의 뭔가를 할때는 매서드를 사용해야함.