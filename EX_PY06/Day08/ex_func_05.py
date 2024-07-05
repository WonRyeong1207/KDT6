# 변수의 데이터가 정해지지 않은 경우
# 매개변수에 전달되는 데이터가 지정되지 않을 경우
# - 어떤 데이터 종류의 값은? dict 형태로 전달하게 됨.
# - 키워드 매개변수 : def 함수명(** params): 키=값의 형태

# 함수 기능 : 회원가입
# 함수 이름 : register
# 매개 변수 : 사바사 **params
# 함수 결과 : result에 받아서 저장

def register(**params):
    if (len(params) > 0):
        print("회원가입")

person = register(name='nana', age=12)

# 함수 기능 : 회원가입
# 함수 이름 : register2
# 매개 변수 : 필수 입력 사항 : id, pw, email
#             선택 입력 사항 : **params
# 함수 결과 : result에 받아서 저장

def register2(id, pw, email, **params):
    pass

register2(id=1232, pw=5454, email=13123)
register2(21, 12, 21, ee=22) # *이 없는 것들은 개수를 맞춰서 입력해야함.
