# Dict 여러타입으로 저장해보기

# 다양한 종류의 키
# - 키가 여러개 정보 합쳐서 사용하는 경우
# - 홍길동 1996, 홍길동 2000
person = {'age':20, ('홍길동', 2000):100} # 키 값은 변경 불가
print(person)

# 3명의 정보를 저장
p1 = {'name':'홍길동', 'age':19, 'job':'학생'}
p2 = {'name':'nana', 'age':20, 'job':'학생'}
p3 = {'name':'kiki', 'age':98, 'job':'majyo'}
p4 = {'name': 'ono_D', 'age':47, 'job':['voice_actocr', 'singer']}

persons_1 = [p1, p2, p3, p4]
print(persons_1[3])

persons_2 = {19:{'name':'홍길동', 'job':'학생'},
             20:{'name':'nana', 'job':'학생'},
             98:{'name':'kiki', 'job':'majyo'},
             47:{'name':'ono_D', 'job':['voice_actor', 'singer', 'radio_DJ']}}
print(persons_2[47])

person_3 = {('홍길동', 19) : {'job' : '학생'},
           ('nana', 20) : {'job':'학생'},
           ('kiki', 98) : {'job':'majyo'},
           ('ono_D', 47) : {'job':['voice_actor', 'singer', 'radio_DJ']}}
print(person_3[('ono_D', 47)])
