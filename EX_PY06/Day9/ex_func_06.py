# 람다 표현식 또는 람다 함수
# - 1줄 함수, 익명 함수
# - 형식 : lamda 매개변수 : 실행코드

names = {1:'Kim', 2:'Adam', 3:'Zoo'}

# 정렬하기 => 내장함수 sorted() -> list 반환
# key로 정렬
result = sorted(names.items())
print("오름차순 정렬 [key]", result)

# value로 정렬
result = sorted(names.items(), key=lambda item:item[1]) # lamda
print("내림차순 정렬 [value]", result)

result = sorted("This is a test string form Andrew.".split())
print(result)

result = sorted("This is test string from Andrew.".split(), key=str.lower)
print(result)

# map()와 lamda
data = [11, 22, 33, 44]

# 각 원소의 값에 곱하기 2해서 다시 리스트 저장
def multi2(value): return value*2

data2 = list(map(multi2, data))
print(data2)

data2 = list(map(lambda x:x*2, data))
print(data2)