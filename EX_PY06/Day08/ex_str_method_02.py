# 문자열에서 좌우 여백 제거 메서드 : strip(), lstrip(), rstrip()
# - 주의 : 문자열 내부의 공백은 제거하지 않음
msg = "Good Luck"
data = "  Happy New Year 2025!   "

# 좌우 모든 공백 제거
m1 = msg.strip()
print(f"origin msg : {len(msg)}, {msg}\nstrip msg : {len(m1)}, {m1}")

d1 = data.strip()
print(f"orign data : {len(data)}, {data}\nstrip data : {len(d1)}, {d1}")
print()

# 외쪽 공백 지우기
m2 = msg.lstrip()
print(f"origin msg : {len(msg)}, {msg}\nlstrip msg : {len(m2)}, {m2}")

d2 = data.lstrip()
print(f"orign data : {len(data)}, {data}\nlstrip data : {len(d2)}, {d2}")
print()

# 오른쪽 겅백 지우기
m3 = msg.rstrip()
print(f"origin msg : {len(msg)}, {msg}\nrstrip msg : {len(m3)}, {m3}")

d3 = data.rstrip()
print(f"orign data : {len(data)}, {data}\nrstrip data : {len(d3)}, {d3}")
print()

# 실습 : 이름을 입력 받아서 저장하세요
# name = input("name : ").strip()
name = 'hin'

if (len(name) > 0):
    print(f"name : {name}")
else:
    print("None name")

# 실습 : 입력받은 데이터에 따라 출력을 다르게 합니다.
# 조건 : 알파벳이면 ★, 숫자면 ♣, 나머지는 무시
data = input("data : ").strip()
data_list = []

for i in range(len(data)):
    if (('a' <= data[i] <= 'z') or ('A' <= data[i] <= 'Z')):
        data_list.append('★')
    elif ('0' <= data[i] <= '9'):
        data_list.append('♣')
    else:
        data_list.append(data[i])
print(''.join(data_list))
print()

