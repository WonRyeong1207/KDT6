# 튜플 내장함수, 연산자, 멤버연산자
# - 내장함수 : len(), max(), min(), sum()
# - 연산자 : 합, 곱, 멤버연산자

nums = 11, 22, 33, 44, 55
print(f"nums의 개수 : {len(nums)}개")
print(f"최댓값 : {max(nums)}, 최솟값 : {min(nums)}")
print(f"합계 : {sum(nums)}")
print(f"정렬 : {sorted(nums)}, {sorted(nums, reverse=True)}")

# 연산자
data1 = 11, 22
data2 = 'A', 'B', 'C'

print(data1+data2)
print(data1*11)

# 멤버연산자 : in, not in
print(f"11 in data1 : {11 in data1}")
print(f"'A' not in data1 : {'A' not in data1}")
