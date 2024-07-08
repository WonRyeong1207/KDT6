# 대화를 할 수 있는 간단한 대화봇
# 미연시처럼 정해진 대답만 할 예정, 단 대답 리스트를 만들어서 대답을 할 것임.
# 일상대화: 일반적인 대화, 비일상대화: 시험기간, 마감기간 대화
# 대답을 랜덤하게 뽑기 위한 라이브러리
import random

# 대딥 리스트
strat_positive_answer_list = ['안녕!', '대화하기를 선택해줘서 고마워.', '어떤 대화를 할까?']
strat_soso_answer_list = ['안녕~', '다음에 다시 만나', 'ㅂㅂ', '나중에 봐']
strat_negative_answer_list = ['잘 보고 입력한거 맞지?', '처음부터 대화할 생각이 없었던 건 아니고?', '허허']






# 입력 데이터 유효성 체크
def input_check(key):
    if len(key) == 1:
        if key.isdecimal():
            return True
        else:
            return False
    else:
        False
        
# test
# d = input("d : ")
# print(input_check(d))

# 입력을 받는 함수
def input_key():
    key = input("대답 (수를 입력) : ")
    if input_check(key):
        key = int(key)
        return key
    else:
        return None

# 초기 화면 함수
def start_ground():
    print(f"{'*':*^100}")
    print(f"* {'선택창':^93} *")
    print(f"{'*':*^100}")
    print(f"* {'1. 대화 시작하기':<91}*")
    print(f"* {'2. 대화 종료하기':<91}*")
    print(f"{'*':*^100}")
    
# 대화를 시작했을때 상태창
def normal_state_background():
    print(f"{'*':*^100}")
    print(f"* {'1. 안녕.':< 80} *")
    print(f"* {'2. 아리야.':<80} *")
    print(f"* {'3. 워하고 있었어?':<80} *")
    print(f"* {'4. 같이 공부하자.':<80} *")
    print(f"* {'5. 같이 놀자!':<80} *")
    print(f"* {'0. 시작화면으로 돌아가기':<80} *")
    print(f"{'*':^100}")

# 대화 리스트 화면을 띄우는 함수
def answer_background(state_num):
    # state_num은 띄우고 싶은 대화 리스트 화면 넘버
    # 초기 화면
    if state_num == 0:
        start_ground()
    
    elif state_num == 1:
        pass
    
    elif state_num == 2:
        normal_state_background()
        
# test
answer_background(0)

answer_key = input_key()
if answer_key == 1:
    print(strat_positive_answer_list[random.randint(0, len(strat_positive_answer_list)-1)],'\n')
elif answer_key == 2:
    print(strat_soso_answer_list[random.randint(0, len(strat_soso_answer_list)-1)], '\n')
else:
    print(strat_negative_answer_list[random.randint(0, len(strat_negative_answer_list)-1)], '\n')

