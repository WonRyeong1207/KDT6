"""
윷놀이 게임
- 2명의 선수 (흥부, 놀부)
- 20점을 먼저 나는 선수가 승리
- 던진 윷가락이 윷이나 모인 경우, 다시 한 번 더 던짐
"""
import time
import random

GOAL = 20
STICK_NUM = 4
player1_name = '흥부'
player2_name = '놀부'

def cast_yut(name, sticks):
    '''
    4개의 윷가락을 던짐: 4개의 랜덤 숫자(0,1)를 생성해서 배열에 추가함
    4개 윷가락(sticks[])의 숫자를 가지고 점수 계산
        숫자             점수
        1111: 모 (horse): 5
        0000: 윷 (cow)  : 4
        0001: 걸 (sheep): 3
        0011: 개 (dog)  : 2
        0111: 도 (pig)  : 1

    :param stick:
    :return:
    '''

    zero_count = 0
    score_names = ['도', '개', '걸', '윷', '모']
    score = 0

    for i in range(STICK_NUM):
        sticks[i] = random.randint(0, 1)
        if sticks[i] == 0:
            zero_count += 1

    if zero_count == 0:  # 모
        score = 5
    else:  # 그 외(도, 개, 걸, 윷)
        score = zero_count

    if name == player1_name:
        print(f'{name} {sticks}: {score_names[score - 1]} ({score}점)', end='')
    else:
        print(f'\t\t\t\t <--- {name} {sticks}: {score_names[score - 1]} ({score}점)', end='')

    return score


def cast_yut_debug(name, sticks):
    '''
        디버깅 용도의 메소드
        :return: 무조건 score=5(모)를 리턴함
    '''
    score_names = ['도', '개', '걸', '윷', '모']
    zero_count = 0

    for i in range(STICK_NUM):
        sticks[i] = 5

    if zero_count == 0:  # 모
        score = 5
    else:  # 그 외(도, 개, 걸, 윷)
        score = zero_count

    if name == player1_name:
        print(f'{name} {sticks}: {score_names[score - 1]} ({score}점)', end='')
    else:
        print(f'\t\t\t\t <--- {name} {sticks}: {score_names[score - 1]} ({score}점)', end='')

    return score


def game_start():

    player1_sticks = [0 for i in range(4)]
    player1_total_score = 0

    player2_sticks = [0 for i in range(4)]
    player2_total_score = 0

    #while player1_total_score < GOAL and player2_total_score < GOAL:
    while True:
        # Player1 수행
        while player1_total_score < GOAL:
            score = cast_yut(player1_name, player1_sticks)
            #score = cast_yut_debug(player1_name, player1_sticks)   #최악의 경우 재현 
            player1_total_score += score
            print(f'/(총 {player1_total_score}점) ---> ')

            if score != 4 and score != 5:
                break

        # player1의 점수가 GOAL 이상이면 경기 종료를 위해
        # 첫 번째 while문 아래 코드 추가
        if player1_total_score >= GOAL:
            break

        # Player2 수행
        while player2_total_score < GOAL:
            score = cast_yut(player2_name, player2_sticks)
            player2_total_score += score
            print(f'/(총 {player2_total_score}점)')
            if score != 4 and score != 5:
                break

        # player2의 점수가 GOAL 이상이면 경기 종료를 위해
        # 첫 번째 while문 아래 코드 추가
        if player2_total_score >= GOAL:
            break


    # 경기 결과 출력
    print('-' * 80)
    if player1_total_score > player2_total_score:
        print(f'{player1_name} 승리 => ', end='')
    elif player1_total_score < player2_total_score:
        print(f'{player2_name} 승리 => ', end='')
    else:
        print(f'무승부=> ', end='')

    print(f'{player1_name}: {player1_total_score} ,{player2_name}:{player2_total_score}')
    print('-' * 80)


game_start()
