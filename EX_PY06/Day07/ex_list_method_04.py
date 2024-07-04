# 리스트의 원소를 제어하기 위한 함수들 메서드

datas = [11, 22, 33]
nums = datas
print(f"datas -> {datas}\nnums -> {nums}\n")

nums[0] = '백'
print(f"datas -> {datas}\nnums -> {nums}")
print()

# 리스트 복사해주는 메서드 : copy() 얕은 복사, 깉은 복사는 모듈을 사용해야함
nums2 = datas.copy()
nums2[0] = 'A'
print(f"datas -> {datas}\nnums2 -> {nums2}\n")

