# 반복문

# 실습 : 문자열을 기계어 즉, 2진수로 변환하여 저장하기
# - 입력 : Hello
# - 출력 : 1010101... 나올듯

msg = 'Hello'
msg = 'a'
result = ''
for i in msg:
    result = result + bin(ord(i))[2:]
    print(result)
# 문자얄 슬라이싱으로 '0b'는 빼고 출력

print()
# 실습 : 요소의 인덱스와 값을 함께 가져오기

nums = [1, 3, 5]
for idx, n in enumerate(nums):
    print(idx, n)
# enumerate() : 전달된 반복가능한 객체에서 원소당 번호를 부여해서 튜플로 묶어줌
#               원소의 인덱스 정보가 필요한 경우 사용
print("enumerate() 변환 : ", list(enumerate(nums)))
