# tuple - 수정 불가, 추가 , 삭제, 변경 불가

nums = 10, 20, 30

# 인덱스 확인 메서드 : index()
idx = nums.index(20)
print(idx)

if 5 in nums:
    idx = nums.index(5)
    print(idx)
    
    
# 데이터 개수 : count()
print(nums.count(10))

