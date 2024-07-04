# dict method
# - keys(), values(), items()

person = {'name' : 'ame', 'age':14}

# get(key) : 값 읽어오는 메서드
print(person['name'])
# print(person['gender']) # key error

print(person.get('name'))
print(person.get('gender', '없음')) # get(key, default) 키에 해당하는 값이 없으면 디폴트 값을 보여줌.

# 키와 값 추가&수정 메서드
person['gander'] = 'man'
print(person)

person.update({'gender':'woman'})
print(person)
person.update(gender='supersuperpeople')
print(person)
person.update(**{'weight':None, 'height':None})
print(person)

person.setdefault('e')
print(person)
person.setdefault('e', 100) # 업데이트가 되지는 않는다
print(person)
person.update(e='@@@@')
print(person)


msg = "Hello Good Luck"
alpha = set(msg)
save = {}

print(alpha)
for m in alpha:
    save[m] = msg.count(m)
print(save)

