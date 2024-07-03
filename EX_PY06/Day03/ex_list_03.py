# 리스트 자료형과 메모리

# 리스트 생성
nums = []
num2 = list()

print(f"nums의 id : {id(nums)}")
print(f"num2의 id : {id(num2)}")
print('\n')

nums = [10, 20]
num2 = list(range(10,30, 10))

print(f"nums의 id : {id(nums)}")
print(f"nums[0]의 id : {id(nums[0])}")
print(f'nums[1]의 id : {id(nums[1])}')
print('\n')
print(f"num2의 id : {id(num2)}")
print(f"num2[0]의 id : {id(num2[0])}")
print(f"num2[1]의 id : {id(num2[1])}")

