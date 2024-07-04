# 리스트 매서드

# 원소 추가 매서드 : append()
datas = [1, 3, 5]

datas.append(6)
print(f"datas 개수 : {len(datas)}, {datas}")

# 원소 추가 메서드 인덱스 사용 : insert()
datas.insert(0, 100) # 0번 위치에 300을 넣어달라
print(f"datas 개수 : {len(datas)}, {datas}")

datas.insert(-1, 20)
print(f"datas 개수 : {len(datas)}, {datas}")
print()

# 실습 : 임의의 정수 숫자 10개 저장하는 리스트 생성
import random

nums = []
range_list = []

for i in range(2):
    range_list.append(random.randint(-100, 100))
range_list = sorted(range_list)

for i in range(10):
    nums.append(random.randint(range_list[0], range_list[1]))
nums = sorted(nums)
print(nums)
print()

# 매서드 원소 삭제 remove()
print(f"datas 개수 : {len(datas)}, {datas}")
datas.remove(100) # 왼쪽에서부터 찾음.
print(f"datas 개수 : {len(datas)}, {datas}")
print()

datas = [100, 29, 40, 29, 100, 59, 299]
for cnt in range(datas.count(100)):
    datas.remove(100)
    print(f"datas 개수 : {len(datas)}, {datas}")
print()

