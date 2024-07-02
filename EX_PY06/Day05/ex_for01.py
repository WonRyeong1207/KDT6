# 반목문
# - 유사하거나 동일한 코드를 1번 작성으로 재상ㅇ하기 위한 방법
# - 종류 : for, while

# for 반목문
# - 시퀀스 데이터에서 원소를 하나씩 읽어올 때 사용
# - 형식 : for 번수명 in 시퀀스:
#          ----xxxxx


# 실습 : 문자열에서 문자열을 1줄에 하나씩 출력하기

msg = 'Merry Christmas 2024'
for i in msg:
    print(i, ord(i), end=' ')

print("END\n")

# 실습 : 리스트에서 원소를 하나씩 읽어오기
# - 1~100 범위에서 3의 배수만 저장한 리스트

nums = []
for i in range(3, 101, 3):
    nums.append(i)
    print(i, end=' ')
print("end\n")
print(f"for문을 이용한 nums : {nums}")

nums = [x for x in range(3, 101, 3)]
print(f"리스트 내포를 이용한 nums : {nums}\n")

# 실습 : str 타입의 원소를 가지는 리스트 입니다.
# 해당 원소를 정수로 형변환 시켜서 저장해주세요
# 워래 원소의 값을 변경해야 하는 경우는 인덱스!!

data = ['4', '9', '1', '3', '8']
print(f"[before] {data}")
data = list(map(int, data))
print(f"[after] {data}")

for i in range(len(data)):
    data[i] = int(data[i])
print(f"[after by for] {data}")

