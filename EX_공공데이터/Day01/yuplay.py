# 윷놀이 게임 프로그램

# 각각 4개의 윷을 던지고 윷이나 모가 나오면 한 번 더 던진다.
# 먼저 20점 점수를 내는 사람이 이김.

# 구현내용
# 윷가락 4개의 값을 저장할 수 있도록 sticks = [0, 0, 0, 0형태로 구현]
# 윷을 던질때마다 랜덤하게 0, 1 사이의 값을 생성해서 sticks[]에 저장하고 점수를 계산
# 예 ) sticks[i] = random.randict(0,1)
# 웇이나 모가 나오면 한번더 던지지만 이미 합이 20점 이상이면 더 던지지 않음
# 시작은 아무나함
# 개임이 종료되면 결과를 출력하고 끝

# 랜덤 모듈 불러오기
import random

# 윷가락 기본값
sticks = [0, 0, 0, 0]

# 기본 설정값
player_1_total = 0
player_2_total = 0
player_1_name = '흥부'
player_2_name = '놀부'

# 윷가락을 던지는 함수
def rad_sticks():
    for i in range(4):
        sticks[i] = random.randint(0, 1)
    return sticks
# test
# sticks = rad_sticks()
# print(sticks)

# 뮻가락의 점수를 구하는 함수
def sum_sticks(sticks_data):
    
    count = 0
    for i in range(len(sticks_data)):
        count = count + sticks_data[i]
        
    if (count == 0):
        return '윷 (4점)', 4
    elif (count == 1):
        return '걸 (3점)', 3
    elif (count == 2):
        return '개 (2점)', 2
    elif (count == 3):
        return '도 (1점)', 1
    else:
        return '모 (5점)', 5
    
# test
# data, score = sum_sticks(rad_sticks())
# print(data, score)

# play 함수를 만들어야 겠구나.
def play(player_name, player_total):
    sticks = rad_sticks()
    data, score = sum_sticks(sticks)
    player_total += score
    
    if player_name == player_1_name:
        print(f"{player_name} {sticks}: {data}/(총 {player_total}점) --->")
    else:
        print(f"                                <--- {player_name} {sticks}: {data}/(총 {player_total}점)")
        
    return score, player_total

# test
# score, p_t = play(player_1_name, player_1_total)

# 메인
while True:
    while True:
        score, player_1_total = play(player_1_name, player_1_total)
        
        if (score == 1) or (score == 2) or (score == 3):
            break
        
    if player_1_total >= 20:
        print('-'*50)
        print(f"{player_1_name} 승리 => {player_1_name} : {player_1_total}, {player_2_name} : {player_2_total}")
        break
    
    while True:
        score, player_2_total = play(player_2_name, player_2_total)
        
        if (score ==1 ) or (score == 2) or (score == 3):
            break
    if player_2_total >= 20:
        print('-'*50)
        print(f"{player_2_name} 승리 => {player_1_name} : {player_1_total}, {player_2_name} : {player_2_total}")
        break