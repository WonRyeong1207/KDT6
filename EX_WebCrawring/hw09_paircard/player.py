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
            for i in range(len(self.open_card_list)):
                print(self.open_card_list[i], end=' ')
            print('\n')
        else:
            print('\n')
            
        print(f"[{self.name}] Holding card list: {len(self.holding_card_list)}")
        for i in range(len(self.holding_card_list)):
            print(self.holding_card_list[i], end=' ')
        print('\n')
    
    def check_one_pair_card(self):
        print(f"[{self.name}: 숫자가 같은 한쌍의 카드 검사]")
        # 문자열로 받는게 나을까? 리스트로 받아서 비교하는 것이 좋을까?
        # 순차 탐색해야하는 거였어...
        
        num_list = []
        for i in range(len(self.holding_card_list)):
            num = str(self.holding_card_list[i])[5]
            num_list.append(num)
        
        # 순차 탐색
        # print(len(self.holding_card_list), len(num_list))
        count = 0
        is_dup_none = False # 중복이 없었음을 알려주는 변수
        while True:
            # 10장 모두가 중복이 아닐 수 있으니까.
            # 종료 조건
            if count > len(num_list):
                break
            
            num_len = len(self.holding_card_list)   # 카드 2장을 빼버리면 수가 줄어드니까.
            
            # 중복을 못 찾는 경우가 발생
            if is_dup_none == True:
                # 이럼에도 중복이 계속 발생하지 않는다면?
                # 여기서 잘 카운트 해야 잘 되는데
                try:
                    current_num = num_list[0+count]
                    n = 1 + count
                except:
                    pass
                    
            else:
                current_num = num_list[0]
                n = 1
                
            
            for j in range(n, num_len): # 왜 같지도 않은데 open으로 빼버리는 거야?
                search_num = num_list[j]
                if current_num == search_num:   # 원소의 값이 다른데 왜 빼는 거야? 왜? 왜?
                    # 같은 것은 보여주려고 추가하고
                    self.open_card_list.append(self.holding_card_list[0])
                    self.open_card_list.append(self.holding_card_list[j])
                    # 그래서 같은 것들은 빼버리고
                    del self.holding_card_list[j]
                    del self.holding_card_list[0]
                    del num_list[j]
                    del num_list[0]
                    # 같으면 찾지도 말라고 사이좋게 제거헸는데
                    is_dup_none = False
                    break   # 같은 원소 찾았으면 추가하고 삭제한 다음에 나오기
                else:
                    is_dup_none = True  # 중복이 없었음을 알려줌
            count += 1
            
                