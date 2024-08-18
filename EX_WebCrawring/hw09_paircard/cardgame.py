# 게임 동작 파일
# - 각 클래스의 객체 생성 및 게임 진행

# 기능
# - 전체 카드 게임을 조율하는 역할
# - 각 클래스를 import 하고 각 클래스의 메소드를 호출해서 게임을 진행
# - Player 객체 생성 및 동작
# - GameDealer의 객체 생성 및 동작

from card import Card
from player import Player
from gamedealer import GameDealer

def play_game():
    # 두 명의 player 객체 생성
    player1 = Player('흥부')
    player2 = Player('놀부')
    
    dealer = GameDealer()
    dealer_deck = dealer.make_deck()
    
    # 카드 각각 20장씩 나누어 주기
    print('='*80)
    print('카드 나누어 주기: 10장')
    print('-'*80)
    
    # 함수를 만들까?
    player1_start_list = []
    player2_start_list = []
    for i in range(10):
        player1_start_list.append(dealer_deck[i])
        player2_start_list.append(dealer_deck[i+1])
        del dealer_deck[i]
        del dealer_deck[i+1]
    player1.add_card_list(player1_start_list)
    player2.add_card_list(player2_start_list)
        
    print(f'[GameDealer] 딜러가 가진 카드 수: {len(dealer_deck)}')
    for i in range(len(dealer_deck)):
        if i in [x for x in range(13, len(dealer_deck), 13)]:
            print()
        print(dealer_deck[i], end=' ')
    print()
    print('='*80)
    player1.display_two_card_list()
    print('='*80)
    player2.display_two_card_list()
    
    _ = input("[2]단계: 다음 단계 진행을 위해 Enter 키를 누르세요!")
    print('='*80)
    player1.check_one_pair_card()
    print('='*80)
    player1.display_two_card_list()
    print('='*80)
    player2.check_one_pair_card()
    print('='*80)
    player2.display_two_card_list()
    
    
if __name__ == '__main__':
    play_game()