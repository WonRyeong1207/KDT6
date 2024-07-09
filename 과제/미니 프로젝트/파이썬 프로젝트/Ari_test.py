# 함수가 너무 많아서 구동만 따로 하려함.
import Ari_func as Ari
import random

# 종료키 입력까지는 유지하고 있어야함. 종료키는 2
while True:
    Ari.answer_background(0)
    answer_key = Ari.input_key()
    
    if answer_key == 1: # 대화시작
        answer_list = Ari.select_list(0, 1)
        talk = Ari.select_answer(answer_list)
        Ari.print_answer(talk)
        
        # 이때도 계속 입력을 받아야함. 종료키는 0
        while True:
            Ari.answer_background(2)
            answer_key = Ari.input_key()
            
            if answer_key == 0: # 시작화면으로 되돌아가기
                answer_list = Ari.select_list(2, 0)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                break
            
            # 대답을 보고 넘어 갈 수 있도록 기다리는 키를 주고 싶음.
            # for문 반복으로 잠시 멈출수 있도록함.
            elif answer_key == 1: # 안녕
                answer_list = Ari.select_list(2, 1)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 2: # 아리야
                answer_list = Ari.select_list(2, 2)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 3: # 뭐하고 있어
                answer_list = Ari.select_list(2, 3)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 4: # 같이 공부하자
                answer_list = Ari.select_list(2, 4)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 5: # 놀자
                answer_list = Ari.select_list(2, 5)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 6: # 힘들었어
                answer_list = Ari.select_list(2, 6)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            else: # 다른 키 입력
                answer_list = Ari.select_list(2)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
    
    elif (answer_key in Ari.byung_list):
        answer_list = Ari.select_list(0, 'byung')
        print('\n')
        for i in range(len(answer_list)):
            print(answer_list[i])
        print('\n')
        for _ in range(100000000): # 경고문은 길어야 하니까. 사람들 급하게 타이핑하다가 긁힘 ㅋ
            pass
        
        while True:
            Ari.answer_background(1)
            answer_key = Ari.input_key()
            
            if answer_key == 0: # 시작화면으로 되돌아가기
                answer_list = Ari.select_list(1, 0)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                break
            
            # 대답을 보고 넘어 갈 수 있도록 기다리는 키를 주고 싶음.
            # for문 반복으로 잠시 멈출수 있도록함.
            elif answer_key == 1: # 안녕
                answer_list = Ari.select_list(1, 1)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 2: # 아리야
                answer_list = Ari.select_list(1, 2)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 3: # 뭐하고 있어
                answer_list = Ari.select_list(1, 3)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 4: # 같이 공부하자
                answer_list = Ari.select_list(1, 4)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 5: # 놀자
                answer_list = Ari.select_list(1, 5)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            elif answer_key == 6: # 힘들었어
                answer_list = Ari.select_list(1, 6)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
                
            else: # 다른 키 입력
                answer_list = Ari.select_list(1)
                talk = Ari.select_answer(answer_list)
                Ari.print_answer(talk)
        
    elif answer_key == 2: # 대화 종료
        answer_list = Ari.select_list(0, 2)
        talk = Ari.select_answer(answer_list)
        Ari.print_answer(talk)
        break
    else: # 다른 키를 입력
        answer_list = Ari.select_list(0)
        talk = Ari.select_answer(answer_list)
        Ari.print_answer(talk)