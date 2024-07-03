# DIct 자료형 살펴보기
# - 네이터의 의미를 함께 저장하는 자료형
# - 형태 : {key1:value1. ...}
# - 키는 중복이 불가능 하지만 값은 중복가능
# - 데이터 분석 시 파일데이터 가져올때 많이 사용

# Dict 생성
data = {}
print(f"data => {len(data)}, {type(data)}, {data}")

# 사람에 대한 정보 : 이름, 나이, 성별
data = {'name':'Ryeong', 'age':24, 'gander':'None'}
print(f"data => {len(data)}개, {type(data)}, {data}")

# 강아지는 잘 몰라서...
cat_Luffy = {'species':'러시안 블루', 'weight':'5.0kg', 'hair_color':'푸른빛의 회색', 'age':'12살'}
print(f"루피의 데이터 : {cat_Luffy}")

# Dict 원소/요소 읽기
# - 키를 가지고 값/데이터 읽기
# - 형식 : 변수명[키]

# 원하는 데이터 출력
print(f"루피의 데이터 : {cat_Luffy['hair_color']}")

# 성별, 품종
print(f"루피의 데이터 \n - 성별 : {cat_Luffy['age']}, 품종 : {cat_Luffy['species']}")

# Dict 데이터 변경
# - 변수명[키] = 새로운 값
# Ryeong의 나이변경
data['age'] = 25
data['species'] = 'half angel, half devil'
print(data)

# Dict 값 삭제
del data['gander']
print(data)

data['age'] = 24
cat_Luffy['name'] = 'Luffy'
# cat_Luffy = cat_Luffy['name', 'species', 'age', 'weight', 'hair_color']
del cat_Luffy['hair_color']
print(f"Ryeong' data\n - name : {data['name']}, {data['age']} \n\nLuffy's data\n - name: {cat_Luffy['name']}")
