# 리스트 전용의 함수, 매소드
# - 리스트의 원소를 데어하기 위한 함수들

# 실습1: 임의의 정수 숫자 10개로 구성된 리스트
import random
nums = []
for i in range(10):
    nums.append(random.randint(1, 100))
print(nums)
print()

# 매서드 - 원소의 인덱스를 반환하는 매서드 index()
data = [7, 3, 9, 11, 5, 7, 2, 1, 3, 4]
idx = data.index(11) # 왼쪽에서 오른쪽으로 찾아감.
print(f"11의 인덱스 : {idx}")

# 존재하지 않는 데이터의 인덱스는 찾을 수 없음.
if 0 in data:
    idx = data.index(0)
    print(f"0의 인덱스 : {idx}")
else: print("0은 존재하지 않는 데이터")

# 중복 데이터는 처음 찾은 데이터의 인덱스 값만 반환
if 3 in data:
    idx = data.index(3)
    print(f"3의 인덱스 : {idx}")
else: print("3은 존재하지 않는 데이터")

if 3 in data:
    idx = data.index(3, 2) # 시작인덱스 위치를 바꿀 수 있음
    print(f"3의 인덱스 : {idx}")
else: print("3은 존재하지 않는 데이터")
print()

# 데이터가 몇개 존재하는지 파악하는 매서드 : count()
cnt = data.count(3)
print(f"3의 개수 : {cnt}")

idx = 0
for i in range(cnt):
    idx = data.index(3, idx+i)
    print(f"3의 인덱스 : {idx}")

'''
index = 0
find_index = []
while (index <= len(data)):
    if 3 in data:
        index = data.index(3, index)
        find_index.append(index)
    index = index + 1
print(find_index)
print()
'''