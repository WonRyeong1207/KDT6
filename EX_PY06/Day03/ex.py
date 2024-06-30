# 내장함수 map()
# - 사용법 : map(자료형, 리스트/튜플, 문자열, 숫자열)
# p.264

# 문자열 입력받음
data = input("숫자 데이터 입력 : ")
print(data, type(data))

# 문자열을 쪼갬
nums = data.split()
print(nums, type(nums))

# 문자열을 정수형으로 변형 - 리스트 내부를 정수로 변환, 리스트 내부 계산은 안됨
int_nums = map(int, nums)
print(int_nums, type(int_nums))

a, b, c, d = map(int, input("숫자를 입력 : ").split())
print(a, b, c, d, type(a))


