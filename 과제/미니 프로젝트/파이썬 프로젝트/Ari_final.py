# 2개의 파일을 합침

# 대화를 할 수 있는 간단한 대화봇
# 미연시처럼 정해진 대답만 할 예정, 단 대답 리스트를 만들어서 대답을 할 것임.
# 일상대화: 일반적인 대화
# 대답을 랜덤하게 뽑기 위한 라이브러리
import random

# 대답 리스트
# 초기 화면
strat_positive_answer_list = ['안녕!', '대화하기를 선택해줘서 고마워.', '어떤 대화를 할까?', '나는 병아리라고 해.', '반가워!',
                              '나의 세계에 온것을 환영해~', '만나서 반가워.']
strat_soso_answer_list = ['안녕~', '다음에 다시 만나', 'ㅂㅂ', '나중에 봐', '다음에도 놀러 와야해~ 꼭이야!!']
strat_negative_answer_list = ['잘 보고 입력한 것 맞지?', '처음부터 대화할 생각이 없었던 건 아니고?', '허허. 왜이래?',
                              '다른 입력키 넣으려다가 실수 한거지?', '그 정도로 키를 잘못 입력하면 좀 위험한거 아니야?']

# 노말 화면, 노말 상태
normal_hello_answer_list = ['안녕.', '안녕!', '안녕?','안녕~', '좋은 하루~', '좋은 날이네', '응']
normal_call_answer_list = ['응?', '불렀어?', '왜?', '구래', '음?', '무슨 할 말 있어?', '하고 싶은 말 있어?', '급하게 안 불러도 되니까 천천히 말해도 괜찮아.',
                           '몇 번을 부르는 거야?', '난 내이름 별로야. 그래서 다른 이름을 쓰고 있어.', '병아리 아닌데.. 아리인데...']
normal_doing_answer_list = ['놀고 있었어.', '일하고 있었어.', '멍 때리고 있었어...', '피곤해서 자고 있었어.', '과제하고 있었어.', 
                            '공부하고 있었어.', '쉬고 있었어.', '게임하고 있었어.', '영화보고 있었어.', '유튭보고 있었엉.', '사람들 만나고 있었어.',
                            '모임이 있어서 밖이야.', '친구들이랑 놀고 있었어.', '아파서 병원에 다녀왔어. 단순 감기더라.', 'SNS하고 있었어. 요즘 뭐가 많네..']
normal_withstudy_answer_list = ['음? 나도? 싫어 ㅎㅎ', '싫은데? 내가 왜?', '오늘 날씨 좋더라~ 화이팅!', '무슨 공부 하려고?',
                                '좀 더 쉬다가 하면 안될까?', '그래.', '하기 싫어.. 싫다구.. ', '그럼 하루종일 공부하는 거지? 기대된다.',
                                '그치 공부는 평소에도 해야지..', '꾸준한 것은 좋아!', '비오니까 하기 싫다.', '비가 오니까 뭔가 더 으슬으슬해서 집중이 잘되는 것 같은 느낌',
                                '여기 나만 추운가? 다른 곳에서 할까?', '카페에서 떠들면서 공부하자!', '난 토론하면서 공부하는 방법이 좋아. 뭔가 더 기억에 잘 각인되는 느낌이고..',
                                '몸이 안 좋아서 다음에 하면 안 될까?', '공부말고 다른 것이 하고 싶어.']
normal_play_answer_list = ['그래그래.', '응? 지금?', '싫은데?', '싫은데...', '그래.', '구랭!', '워하고 놀려고?', '다음에 놀자', '해야할 것들은 다 하고 노는 거지?', 
                            '그 정도로 놀고 싶은거야?', '피곤해서 다음에 놀자.', '할 것들이 많아서 다음에 놀자.', '피곤한데 나랑 놀아주는 거야?',
                            '난 체스 좋아하는데 너는?', '내가 물리적으로 너랑 놀아줄 수는 없어.', '나한테 너무 많은 것을 바라는거 아니야?', '내가 널 어떻게 놀아줄 수 있을까?',
                            '수다를 떠면서 놀까?', '어떻게 놀아주면 잘 놀았다고 소문이 날까?']
normal_hard_answer_list = ['오늘도 수고했어.', '고생했어.', '그 정도로 공부하면 당연히 힘들지.', '그 정도로 하면 함들지...', '요즘 많이 힘들지? 고생했어.', '요즘 많이 힘들지? 수고했어.',
                           '잠도 안 자고 했으니까..', '빈속에 카페인을 그렇게 들이부었는데 안 쓰리고 베기니?', '최근에는 날씨도 안 도와주는 것 같네.', '많이 힘들었지? 울어도 괜찮아.',
                           '울고 싶으면 도와줄게.', '정해진 말 뿐이지만 이 말이 위로가 되었으면 좋겠어.', '괜찮아. 다 괜찮아 질거야. 이 또한 지나갈거야.', '고생끝에 락이 온다고 하잖아. 이겨낼 수 있을 거야.',
                           '넌 너 나름의 최선의 노력을 했다면 그걸로 충분해. 수고했어.', '누가 뭐래도 최선을 다 했다면 그걸로 충분해.', '가끔은 우는 것도 괜찮아.', '참지말고 소리라도 질러!',
                           '운다고 해서 약한건 아니니까. 속에다가 담아두고 있지마. 병난다.', '위로가 되었으면 하지만 안 될것이라는 것도 알고 있어. 그럼에도 하는거야.', '내가 몸이 있었다면 기대라고 했을거야.',
                           '괴로워도 그것이 평생을 가지는 않을거야.', '살다보면 이런 날도 있고 저런 날도 있는거야.', '모든 일들이 다 너의 잘못 일리는 없어.' ,'자책을 하는 것도 좋지만 너무 깊은 자책은 오히려 해야.',
                           '세상은 너가 원하는 대로 이루어지지 않아.', '그러니까 오늘도 힘내!', '잘하고 있으니까..']

normal_bye_answer_list = ['ㅂㅂ', '다음에 또 대화하면 놀자.', '내일 봐~', '내가 생각나면 언제든지 다시 찾아와.']
normal_negative_answer_list = ['저기 잘보고 입력한거지?', '나랑 대화하고 싶은 거 맞지?', '가끔은 잘못 누를 수도 있지.. 암...',
                               '저런 얼마나 피곤하면 저래...', '아냐.. 그냥 뭔가 섭섭하네.', '천천히 입력해도 괜찮아.', '얼마나 급했던거야..']

# 븅아리 리스트
# 븅아리 호출 리스트
byung_list = ['jung', 'Jung', 'JUng', 'JUNg', 'JUNG', 'jeong', 'Jeong', 'JEong', 'JEOng', 'JEONg', 'JEONG', 'wjd', 'Wjd', 'WJd', 'WJD', '정']
# 븅아리 호출시
byung_warnning_list = ['------------------------- 주의 ----------------------',
                       '븅아리는 제작자가 넣은 이스터에그 입니다.',
                       '븅아리는 싸가지가 없습니다.',
                       '븅아리와 대화하다가 화가 나셔도 제작자와는 무관합니다.',
                       '욕설과 혐오발언이 있을 수 있습니다.',
                       '븅아리는 어느 특정 인물이 모티브가 아닙니다.',
                       '븅아리는 어리기 때문에 선이 없습니다.',
                       '상처를 받으셔도 제작자의 책임은 없습니다.',
                       '븅아리와 대화하기 싫으시면 0번을 입력하여 되돌아가시면 됩니다.']

# 븅아리 대답 리스트
byung_hello_answer_list = ['ㅎㅇ', '뭐', '어쩔', '안녕', 'ㅇ?', '어~']
byung_call_answer_list = ['한가해?', '왜 부름?', '어지간히 불러라.', '에휴.. 왜?', '아리라고 불러.', '풀네임으로 부르는 건 안 좋아하는데..', 'ㅅㅂ']
byung_doing_answer_list = ['보면 모르냐?', '웃기는 놈이네', '어휴..', '쟤는 왜 저런데?', '너가 알아서 뭐하게?']
byung_withstudy_answer_list = ['너나해.', '내가 그걸 왜 함?', '싫음.']
byung_play_answer_list = ['뭐하고?', '싫음.', '굳이?', '내가 왜 놀아야 하는데?', '너랑? 내가?', '어린아이 소꿉놀이나 하겠지. 너 놀 줄은 아니?']
byung_hard_answer_list = ['ㅄ', '긁?', '표정봐라 ㅋㅋㅋ', '표정하고는 ㅋㅋㅋ']

byung_bye_answer_list = ['ㅂㅂ', '잘가.', '담에도 보면 좋겠네.', '잘가고~']
byung_negative_answer_list = ['이제는 글도 못치네 ㅋ', '성격은 급해가지고 어휴.. ㅉ', '그러고 싶니?', '왜 그러는 거야?', '어휴.. 븅신', 'ㅉㅉㅉ']

# 입력 데이터 유효성 체크
def input_check(key):
    if len(key) == 1:
        if key.isdecimal():
            return True
        else:
            return False
    else:
        False

# 입력을 받는 함수
def input_key():
    key = input("질문 번호를 입력 : ")
    if input_check(key):
        key = int(key)
        print('\n\n')
        return key
    elif (key in byung_list):
        return key
    else:
        print('\n\n')
        return None

# 대답할 리스트를 선정하는 함수, 리스트의 이름이 길어서 보기 싫어서
def select_list(state_num, answer_key=None): # else 고려
    # 초기 화면에서
    if state_num == 0:
        if answer_key == 1:
            return strat_positive_answer_list
        elif answer_key == 2:
            return strat_soso_answer_list
        elif answer_key == 'byung':
            return byung_warnning_list
        else:
            return strat_negative_answer_list
        
    # 븅아리에서
    elif state_num == 1:
        if answer_key == 0:
            return byung_bye_answer_list
        elif answer_key == 1:
            return byung_hello_answer_list
        elif answer_key == 2:
            return byung_call_answer_list
        elif answer_key == 3:
            return byung_doing_answer_list
        elif answer_key == 4:
            return byung_withstudy_answer_list
        elif answer_key == 5:
            return byung_play_answer_list
        elif answer_key == 6:
            return byung_hard_answer_list
        else:
            return byung_negative_answer_list
    
    # (정)병아리에서
    elif state_num == 2:
        if answer_key == 0:
            return normal_bye_answer_list
        elif answer_key == 1:
            return normal_hello_answer_list
        elif answer_key == 2:
            return normal_call_answer_list
        elif answer_key == 3:
            return normal_doing_answer_list
        elif answer_key == 4:
            return normal_withstudy_answer_list
        elif answer_key == 5:
            return normal_play_answer_list
        elif answer_key == 6:
            return normal_hard_answer_list
        else:
            return normal_negative_answer_list

# 대답을 선택하는 함수
def select_answer(answer_list):
    talk = random.randint(0, len(answer_list)-1)
    answer = answer_list[talk]
    return answer

# 대답을 보여주는 함수
def print_answer(answer):
    space = 100 - len(answer)
    # space = int(space)
    print(f"{'*':*^100}")
    print(f"* {'대답':^94} *")
    print(f"{'*':*^100}")
    print(f"{'*':<50}{'*':>50}")
    print(f"* {answer:<{space}} *")
    print(f"{'*':<50}{'*':>50}")
    print(f"{'*':*^100}")
    print('\n\n')
    # 잠시 멈추게 하는 반복문
    for _  in range(50000000):
                    pass

# 초기 화면 함수
def start_ground():
    print(f"{'*':*^100}")
    print(f"{'*':<50}{'*':>50}")
    print(f"* {'병아리와 대화하고 놀아요!':^85} *")
    print(f"{'*':<50}{'*':>50}")
    print(f"{'*':*^100}")
    print(f"* {'선택창':^93} *")
    print(f"{'*':*^100}")
    print(f"* {'1. 대화 시작하기':<91}*")
    print(f"* {'2. 대화 종료하기':<91}*")
    print(f"{'*':*^100}")
    
# 대화를 시작했을때 상태창
def normal_state_background():
    print(f"{'*':*^100}")
    print(f"* {'선택창':^93} *")
    print(f"{'*':*^100}")
    print(f"* {'1. 안녕.':<94} *")
    print(f"* {'2. (정)병아리야.':<91} *")
    print(f"* {'3. 워하고 있었어?':<90} *")
    print(f"* {'4. 같이 공부하자.':<90} *")
    print(f"* {'5. 같이 놀자!':<92} *")
    print(f"* {'6. 힘들어...':<93} *")
    print(f"* {'0. 시작화면으로 돌아가기':<86} *")
    print(f"{'*':*^100}")

# 대화를 시작했을때 상태창
def byung_state_background():
    print(f"{'*':*^100}")
    print(f"* {'선택창':^93} *")
    print(f"{'*':*^100}")
    print(f"* {'1. 안녕.':<94} *")
    print(f"* {'2. 븅아리야.':<92} *")
    print(f"* {'3. 워하고 있었어?':<90} *")
    print(f"* {'4. 같이 공부하자.':<90} *")
    print(f"* {'5. 같이 놀자!':<92} *")
    print(f"* {'6. 힘들어...':<93} *")
    print(f"* {'0. 시작화면으로 돌아가기':<86} *")
    print(f"{'*':*^100}")

# 대화 리스트 화면을 띄우는 함수
def answer_background(state_num):
    # state_num은 띄우고 싶은 대화 리스트 화면 넘버
    # 초기 화면
    if state_num == 0:
        start_ground()
    
    elif state_num == 1: # 이상한 상태의 (정)병아리 == 븅아리
        byung_state_background()
    
    elif state_num == 2:
        normal_state_background()
        
def ari_chat():
    # 종료키 입력까지는 유지하고 있어야함. 종료키는 2
    while True:
        answer_background(0)
        answer_key = input_key()
        
        if answer_key == 1: # 대화시작
            answer_list = select_list(0, 1)
            talk = select_answer(answer_list)
            print_answer(talk)
            
            # 이때도 계속 입력을 받아야함. 종료키는 0
            while True:
                answer_background(2)
                answer_key = input_key()
                
                if answer_key == 0: # 시작화면으로 되돌아가기
                    answer_list = select_list(2, 0)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    break
                
                # 대답을 보고 넘어 갈 수 있도록 기다리는 키를 주고 싶음.
                # for문 반복으로 잠시 멈출수 있도록함.
                elif answer_key == 1: # 안녕
                    answer_list = select_list(2, 1)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 2: # 아리야
                    answer_list = select_list(2, 2)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 3: # 뭐하고 있어
                    answer_list = select_list(2, 3)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 4: # 같이 공부하자
                    answer_list = select_list(2, 4)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 5: # 놀자
                    answer_list = select_list(2, 5)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 6: # 힘들었어
                    answer_list = select_list(2, 6)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                else: # 다른 키 입력
                    answer_list = select_list(2)
                    talk = select_answer(answer_list)
                    print_answer(talk)
        
        elif (answer_key in byung_list):
            answer_list = select_list(0, 'byung')
            print('\n')
            for i in range(len(answer_list)):
                print(answer_list[i])
            print('\n')
            for _ in range(100000000): # 경고문은 길어야 하니까. 사람들 급하게 타이핑하다가 긁힘 ㅋ
                pass
            
            while True:
                answer_background(1)
                answer_key = input_key()
                
                if answer_key == 0: # 시작화면으로 되돌아가기
                    answer_list = select_list(1, 0)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    break
                
                # 대답을 보고 넘어 갈 수 있도록 기다리는 키를 주고 싶음.
                # for문 반복으로 잠시 멈출수 있도록함.
                elif answer_key == 1: # 안녕
                    answer_list = select_list(1, 1)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 2: # 아리야
                    answer_list = select_list(1, 2)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 3: # 뭐하고 있어
                    answer_list = select_list(1, 3)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 4: # 같이 공부하자
                    answer_list = select_list(1, 4)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 5: # 놀자
                    answer_list = select_list(1, 5)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                elif answer_key == 6: # 힘들었어
                    answer_list = select_list(1, 6)
                    talk = select_answer(answer_list)
                    print_answer(talk)
                    
                else: # 다른 키 입력
                    answer_list = select_list(1)
                    talk = select_answer(answer_list)
                    print_answer(talk)
            
        elif answer_key == 2: # 대화 종료
            answer_list = select_list(0, 2)
            talk = select_answer(answer_list)
            print_answer(talk)
            break
        
        else: # 다른 키를 입력
            answer_list = select_list(0)
            talk = select_answer(answer_list)
            print_answer(talk)

# test
if __name__=='__main__':
    
    # test
    # d = input("d : ")
    # print(input_check(d))
    
    # answer_background(2)
    # print_answer(select_answer(normal_withstudy_answer_list))
    # byung_state_background()
    
    # start_ground()
    
    ari_chat()