# 전역변수
persons = ['hong']
gender = {'hong':'man'}
year = 2024

# 사용자 정의 함수
def add_person(name):
    global year
    year += 1
    persons.append(name)
    gender[name] = 'woman'
    
def remove_person(name):
    persons.remove(name)
    gender.pop(name)
    
# 함수  출력
print(f"person => {persons}")

add_person('park')
print(f"person => {persons}")
print(f"gender => {gender}")

remove_person('park')
print(f"person => {persons}")
print(f"gender => {gender}")
