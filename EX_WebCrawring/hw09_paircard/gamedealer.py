# GameDealer 클래스
# - 1벌의 카드(deck) 생성 기능: 리스트로 구현
# - 각 Player들에게 카드를 나누어 주는 기능
#   - 자신이 가진 deck에서 제거하여 다른 선수들에게 제공

# GameDealer 클래스 기능
# - GameDealer 객체는 card_suit, card_number를 이용하여 Card 객체 생성 및 리스트(deck)에 저장
#   - card_suit = ["♠",	"♥", "♣", "◆"]
#   - card_number = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

from card import Card
import random

class GameDealer:
    def __init__(self):
        self.deck = list()
        self.suit_number = 13
        
    def make_deck(self):
        card_suit = ["♠", "♥", "♣", "◆"]
        card_number = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        
        # 여기서부터 구현
        # 리스트의 deck의 값을 랜덤하게 섞음
        # - random.shufle(리스트) 함수 호출
        # - 랜덤하게 섞인 deck 내용 출력: 위의 deck 내용 출혁함수 재사용
        
        # 초기 출력 내용
        print('[GameDealer] 초기 카드 생성')
        print('-'*80)
        print(f'[GameDealer] 딜러가 가진 카드 수: {len(card_suit)*len(card_number)}')
        for suit in card_suit:
            for number in card_number:
                card = Card(suit, number)
                self.deck.append(card)
                print(card, end=' ')
            print()
        print()
        
        # 랜덤하게 섞기
        random.shuffle(self.deck)
        print('[GameDealer] 카드 랜덤하게 섞기')
        print('-'*80)
        print(f"[GameDealer] 딜러가 가진 카드 수: {len(self.deck)}")
        for i in range(len(self.deck)):
            if i in [x for x in range(13, len(self.deck), 13)]:
                print()
            print(self.deck[i], end=' ')
        print()
    
        return self.deck
    
    
if __name__ == '__main__':
    
    dealer = GameDealer()
    deck = dealer.make_deck()
    print('\n\n')
    i = 0
    for card in deck:
        card_num = str(card)[5]
        print(card_num, end=' ')
        if i in [x for x in range(12, len(deck), 13)]:
            print()
        i += 1