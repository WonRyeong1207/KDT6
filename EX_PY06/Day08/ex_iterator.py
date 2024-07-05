# Iteraotr 객체 : 반복자를 가지고 있는 객체
# - 커스텀 클래스 생성 확인
# - 이미 Iterator 객체를 가지고 있는 객체 살펴보기

# 내장함수 dir() : 객체가 가지는 변수와 메서드를 리스트로 알려줌
nums = [11, 33, 55]

# _ 변수 : 의미없는 변수 선언, 문법상 필요한 변수 선언
'''
for _ in dir(nums):
    print(_)
print()
# __iter__ 가 있어서 반복 가능
'''

# 리스트에서 반복자(Iterator) 추출 : .__iter__()
iterator = nums.__iter__()

print(iterator.__next__())
print(iterator.__next__())
print(iterator.__next__())
print()

# 내장함수 iter(반복이 가능한 객체)
iterator = iter(nums) # 객체에 존재하는 반복자 추출
print(iterator.__next__())
print(iterator.__next__())
print(iterator.__next__())
# 메모리때문에 사용함
