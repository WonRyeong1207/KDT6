# dict method
# - keys(), values(), items()

person = {'name' : 'ame', 'age':14}

# get(key) : 값 읽어오는 메서드
print(person['name'])
# print(person['gender']) # key error

print(person.get('name'))
print(person.get('gender', '없음')) # get(key, default) 키에 해당하는 값이 없으면 디폴트 값을 보여줌.

