# Dict : 연산자와 내장함수

cat = {'name':'Luffy', 'species':'러시안 블루', 'weight':'5.0kg', 'hair_color':'푸른빛의 회색', 'age':'12살'}
person = {'name': 'ono_D', 'age':47, 'job':['voice_actocr', 'singer', 'radio_DJ']}

# 연산자
# - 산술연산 X
# person + cat
# print(person + cat)

# 멤버 연상자 in, not in
print('name' in cat)
print('name' in person)

# value : dict 타입에서는 key만 멤버 연산자로 확인
print('러시안 블루' in cat.values())
print('singer' in person.values())

# 내징힘수
# - 원소/요소 개수 확인 : len()
print(f"cat의 원소 개수 : {len(cat)}")
print(f"person의 원소 개수 : {len(person)}")

# - 원소/요소 정렬 : sorted()
# 키를 기준으로 정렬
print(f"cat 정렬 : {sorted(cat)}")
print(f"cat 정렬 : {sorted(cat.values())}") # 값의 type이 여러개라서 비교 불가, 같으면 정렬가능!
print(f"person 정렬 : {sorted(person, reverse=True)}")

score = {'korean' : 90, 'math':80, 'english':80, 'sport':60}
print(f" 점수의 정렬 : {sorted(score)}")
print(f"점수의 정렬 : {sorted(score.values())}")
print(f"점수의 정렬 : {sorted(score.items())}")
print(f"점수의 정렬 : {sorted(score.items(), key=lambda x:x[1])}") # x의 값을 바꿔서 함수를 진행
