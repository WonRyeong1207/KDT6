# kdt6 황지원
# 클래스를 이용한 커피 자판기 프로그램 설계 및 구현

"""
 모든 커피의 가격은 300원으로 동일하며 선택한 메뉴에 따라 각 물품의 소모량은 아래와 같이
소모가 됨.
 충분한 돈을 입력하지 않았거나, 자판기 내부의 물품(커피, 프림, 설탕, 물, 종이컵 등)이
부족한 경우에는 재료가 부족하다는 메시지를 출력하고 프로그램을 종료함.
 사용자가 선택한 커피 메뉴에 따라 각 물품의 소모량은 다르며, 커피를 제공하면 선택 메뉴에 따른
재료 현황(딕셔너리 형태)을 업데이트 하며, 종료 메뉴를 선택하면 남은 돈을 반환함.
 각 기능들은 클래스의 메소드로 구현하고, 필요한 메소드는 추가 구현하면 됨.
 
■ 초기 자판기 재료 현황: 아래 물품은 모두 딕셔너리로 구현
 - 자판기 내부 동전: 0원 (사용자가 커피를 먹을 때 마다 300원씩 증가)
 - 물: 500ml
 - 커피: 100g, 프림: 100g, 설탕: 100g, 종이컵 5개
 - 딕셔너리는 클래스 생성자에 선언 후 각 함수들에서 사용
 
■ 메뉴 출력 및 선택 기능 (종료를 선택할 때까지 반복)
 - 초기에 1회만 동전을 투입하고 메뉴를 선택
 - 투입된 돈이 300원 이상인 경우에만 커피를 제공
 - 메뉴 출력시 현재 잔액을 화면에 표시
 - 메뉴 1. 블랙 커피, 2. 프림 커피, 3: 설탕 프림 커피, 4. 재료 현황, 5. 종료

■ 커피 제공 기능: 메뉴에 따른 커피, 설탕, 프림 소모량
 - 먼저 자판기에 남은 재료의 양을 검사한 다음, 선택한 메뉴에 따라 충분한 재료가
   남아 있는 경우에 한해서 커피를 제공하며 커피를 제공한 다음 재료 현황을 업데이트
   하고 화면에 출력
 - 블랙 커피: 커피 30g + 물 100ml
 - 프림 커피: 커피 15g + 프림 15g + 물 100ml
 - 설탕 프림 커피: 커피 10g + 프림 15g + 설탕 10g + 물 100ml

■ 재료 현황 업데이트 기능: 딕셔너리 업데이트
 - 커피, 프림, 설탕, 컵, 잔여 물 용량 업데이트

■ 재료 현황 출력 기능
 - 커피를 제공하면 현재 자판기에 남아 있는 커피량, 프림량, 설탕량, 컵의 개수,
   남은 물 용량 출력

■ 물품 현황 체크 기능
 - 사용자가 선택한 커피 메뉴에 필요한 물품 현황 체크
 - 충분한 물품이 없는 경우, '재료가 부족합니다.' 출력하고 남은 돈을 반환한 다음 프로그램 종료
"""


# 주어진 형태
class VendingMachine:
    def __init__(self, input_dict):
        '''
        생성자
        :param input_dict: 초기 자판기 재료량(dict형태)
        '''
        self.input_money = 0
        self.inventory = input_dict
    
    # 입력 데이터 유효성 체크
    def money_check(self, money):
        self.money = money
        if len(self.money) <= 4:       # 최대 판매할 수 있는 커피 값이 1500원
            if self.money.isdecimal():
                return True
            else:
                return False
        else:
            return False

    # 입력을 받는 함수
    def input_money_(self):
        money = input("동전을 투입하세요 : ")
        if self.money_check(money):
            money = int(money)
            return money
        else:
            return None
    
    # 입력 데이터 유효성 체크
    def key_check(self, key):
        self.key = key
        if len(self.key) == 1:
            if self.key.isdecimal():
                return True
            else:
                return False
        else:
            return False

    # 입력을 받는 함수
    def input_key(self):
        key = input("메뉴를 선택하세요 : ")
        if self.key_check(key):
            key = int(key)
            return key
        else:
            return None
    
    
    def run(self):
        '''
        커피 자판기 동작 및 메뉴 호출 함수
        '''
        # 기능 구현 및 다른 메서드 호출
        
        
        while True:
            # 돈을 입력받음
            self.input_money = self.input_money_()
            
            # None인 경우를 먼저 검증하면 에러 안나려나?
            if self.input_money == None:
                print('-'*80)
                print("커피 자판기 동작을 종료합니다.")
                print('-'*80)
                break
            
            # 300원 이상인 경우
            elif self.input_money >= 300:
                pass
            
            # 300원 보다 작을 경우
            elif self.input_money < 300:
                print(f"투입된 돈 ({self.input_money}원)이 300원 보다 작습니다.")
                print('-'*80)
                print("커피 자판기 동작을 종료합니다.")
                print('-'*80)
                break
            
            # 만약의 경우..
            else:
                print('Error')
                print('-'*80)
                print("커피 자판기 동작을 종료합니다.")
                print('-'*80)
                break
        
        
if __name__ == '__main__':
    # VendingMachine 객체 생성
    inventory_dict = {'coffee':100, 'cream':100, 'sugar':100,
                    'water':500, 'cup':5, 'change':0}
    coffee_machine = VendingMachine(inventory_dict)
    coffee_machine.run()    # VendingMachine 동작 메서드