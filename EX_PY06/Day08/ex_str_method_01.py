# 문자열 메서드

msg = "Hello 0705"

# 원소 인덱스 찾기 메서드 : .find(문자 1개 또는 문자열)
# - 'H'의 인덱스
idx = msg.find("H") # 왼쪽에서부터 찾아줌
print(f"'H'의 인덱스 : {idx}\n")

idx = msg.find('l') # 먼저 찾은 인덱스만 반환함
print(f"'l'의 인덱스 : {idx}\n")

idx = msg.find('llo') # 문자열의 시작위치를 반환
print(f"'llo'의 시작 인덱스 : {idx}\n")

idx = msg.find('llO')
print(f"'llO'의 인덱스 : {idx}\n") # 없으면 -1을 결과로 줌.

# 원소의 인덱스 찾기 메서드 : .index(문자1개 또는 문자열)
idx = msg.index("H")
print(f"H의 인덱스 : {idx}\n")

# idx = msg.index('h') # 존재하지 않기에 에러를 발생

if ('h' in msg):
    idx = msg.index('h')
    print(f"h의 인덱스 : {idx}\n")
else:
    print("h는 msg에 존재하지 않음\n")

# 문자열에 동일한 문자가 존재하는 경우
msg = "Good Luck Good"
# - o의 인덱스를 찾고 싶음
idx = msg.index('o')
print(f"o의 인덱스 : {idx}")
# - o의 두번째 인덱스
idx = msg.index('o', idx+1)
print(f"o의 인덱스 : {idx}")
print()

# 반복문을 이용
cnt = msg.count('o')
idx = 0
for i in range(cnt):
    idx = msg.index('o', idx+1)
    print(f"o의 {i+1}번째 인덱스 : {idx}")
print()

# 문자열의 뒷부분부터 찾기하는 메서드 : rfind(), rindex()
msg = "Happy"

# - y 인덱스 찾기
idx = msg.rfind('y')
print(f"y index: {idx}")
idx = msg.rfind('H')
print(f"H index : {idx}")

idx = msg.rfind('p')
print(f"p index : {idx}")
idx = msg.rfind('p', 0, idx) # 시작과 끝을 지정. 끝의 넣은 수는 포함하지 않음. [a, b)
print(f"p index : {idx}")
print()

# 파일 확장자의 경우 확장자는 뒤에서 찾는게 빠를 수 있음.
files = ['hello.txt', '202년상반기경제분석.doc', 'kakao_123456789.jpg']
file_ext_list = []

for i in files:
    idx = i.rfind('.')
    file_ext_list.append(i[idx:])
print(file_ext_list)
print()

