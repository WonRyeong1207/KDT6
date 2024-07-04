# 메서드 원소 순서 제어 메서드 : reverse()
import random
random.seed(10) # 동일한 랜덤 숫자 추출을 위한 기준점

datas = [random.randint(1, 30) for i in range(10)]
print(f"{len(datas)}개, {datas}\n")

# 0번을 -1, -1번을 0번으로 위치 변경
datas.reverse()
print(f"{len(datas)}개, {datas}\n")

# 원소의 크기를 비교해서 정렬해주는 메서드 : sort()
# - 기본은 오름차순
datas.sort()
print(f"{len(datas)}개, {datas}\n")

datas.sort(reverse=True) # 내림차순
print(f"{len(datas)}개, {datas}\n")

# 리스트에서 원소를 꺼내는(삭제) 메서드 : pop()
# - 리스트에서 원소를 삭제, 변수가 있다면 그 값을 받음
value = datas.pop() # 제일 마지막 원소를 꺼냄
print(f"value : {value} - {len(datas)}개, {datas}\n")

value = datas.pop(0) # 원하는 인덱스의 원소 뽑아옴
print(f"value : {value} - {len(datas)}개, {datas}\n")

# 리스트를 확장시켜주는 메서드 : extand()
datas.extend([11, 22, 33])
print(f"{len(datas)}개, {datas}\n")

datas.extend("Good Luck") # 연속적인 데이터 형은 전부 다 가능 , str, set, tuple, list, dict
print(f"{len(datas)}개, {datas}\n")

datas.extend({55, 77, 11, 22, 77})
print(f"{len(datas)}개, {datas}\n")

datas.extend({'name':'nana', 'age':12})
print(f"{len(datas)}개, {datas}\n") # key 만 들어감

# 모든 원소 삭제 : clear()
datas.clear()
print(f"{len(datas)}개, {datas}\n")

