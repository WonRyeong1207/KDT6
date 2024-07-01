# Dict 전용함수 (매서드)

# Dict에서 키만 추출 : Keys()
p4 = {'name': 'ono_D', 'age':47, 'job':['voice_actocr', 'singer', 'radio_DJ']}
print(f"person_4 Key\n - {p4.keys()}, {type(p4.keys())}")

# Dict에서 값만 추출 : values()
print(f"person_4 Value\n - {p4.values()}, {type(p4.keys())}")
print(f"키와 값 묶음 추출 : {p4.items()}, {type(p4.items())}")
# Dict_xxx 이니까 추출해서 사용할 것이라면 형변환을 해야함.
