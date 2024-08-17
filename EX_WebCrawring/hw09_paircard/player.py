# Player 클래스
# - 자신이 가지고 있는 카드 관리
#   - 두 개의 리스트를 가짐 (holding_card_list, open_card_list)
# - 두 장의 동일한 카드를 제거하는 기능
#   - 번호가 동일한 경우, holding_card_list에서 open_card_list로 이동
#       : 테이블에 공개하는 기능
# - 두 개의 리스트를 출력하는 기능

class Player:
    def __init__(self, name):
        self.name = name
        self.holding_card_list = list()
        self.open_card_list = list()
        
    def add_card_list(self, card_list):
        for card in card_list:
            self.holding_card_list.append(card)
    
    def display_two_card_list(self):
        print(f'[{self.name}] Open Card list: {len(self.open_card_list)}')
        
        if len(self.open_card_list) > 0:
            for i in range(2):
                print(self.open_card_list[i], end=' ')
            print()
        else:
            print()
    
    def check_one_pair_card(self):
        card_num_list = []
        # 문자열로 받는게 나을까? 리스트로 받아서 비교하는 것이 좋을까?
        for i in range(len(self.holding_card_list)):
            card = self.holding_card_list[i]
            if (i == 0) or (i == len(self.holding_card_list)):    
                pass
            elif (i >= 1) and (i < len(self.holding_card_list)):
                card_num = str(card)[5]
                next_num = str(self.holding_card_list[i+1])[5]
                if card_num == next_num:
                    self.open_card_list.append(self.holding_card_list[i])
                    self.open_card_list.append(self.holding_card_list[i+1])
                    del self.holding_card_list[i+1]
                    del self.holding_card_list[i]
                    break
                
                