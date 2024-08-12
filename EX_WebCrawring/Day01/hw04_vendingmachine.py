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
        if len(self.money) <= 5:       # 설마 동전을 십만원 단위로 넣지는 않겠지?
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
    
    # 메뉴를 띄우는 함수
    def menu(self):
        print('1. 블랙 커피')
        print('2. 프림 커피')
        print('3. 설탕 프림 커피')
        print('4. 재료 현황')
        print('5. 종료')
    
    # 업데이크 함수
    def update(self, coffee, cream, sugar, water, cup, change, money):
        self.inventory['coffee'] = coffee
        self.inventory['cream'] = cream
        self.inventory['sugar'] = sugar
        self.inventory['water'] = water
        self.inventory['cup'] = cup
        self.inventory['change'] = change
        self.input_money = money
    
    # 블랙 커피
    def black_coffee(self):
        # 변수부터 너무 기니까 정리
        coffee = self.inventory['coffee']
        cream = self.inventory['cream']
        sugar = self.inventory['sugar']
        water = self.inventory['water']
        cup = self.inventory['cup']
        change = self.inventory['change']
        money = self.input_money
        
        # 재료 소진부터 체크
        if (coffee < 30) or (water < 100) or (cup < 0):
            print('재료가 부족합니다.')
            print('-'*80)
            print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
            print('-'*80)
            print(f"{money}원을 반환합니다.")
            print('-'*30)
            print("커피 자판기 동작을 종료합니다.")
            print('-'*30)
            stop = True
            return stop
        
        # 커피 판매
        money = money - 300
        coffee = coffee - 30
        water = water - 100
        cup = cup - 1
        change = change + 300
        print(f"블랙 커피를 선택하셨습니다. 잔액: {money}")
        print('-'*80)
        print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
        print('-'*80)
        
        # 커피 업데이트??
        self.update(coffee, cream, sugar, water, cup, change, money)
        
        stop = False
        return stop
        
    # 프림 커피
    def cream_coffee(self):
        coffee = self.inventory['coffee']
        cream = self.inventory['cream']
        sugar = self.inventory['sugar']
        water = self.inventory['water']
        cup = self.inventory['cup']
        change = self.inventory['change']
        money = self.input_money
        
        # 재료 소진부터 체크
        if (coffee < 15) or (water < 100) or (cup < 0) or (cream < 15):
            print('재료가 부족합니다.')
            print('-'*80)
            print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
            print('-'*80)
            print(f"{money}원을 반환합니다.")
            print('-'*30)
            print("커피 자판기 동작을 종료합니다.")
            print('-'*30)
            stop = True
            return stop
        
        # 커피 판매
        money = money - 300
        coffee = coffee - 15
        water = water - 100
        cup = cup - 1
        change = change + 300
        cream = cream - 15
        print(f"프림 커피를 선택하셨습니다. 잔액: {money}")
        print('-'*80)
        print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
        print('-'*80)
        self.update(coffee, cream, sugar, water, cup, change, money)
        stop = False
        return stop
    
    
    # 설탕 프림 커피
    def sugar_cream_coffee(self):
        coffee = self.inventory['coffee']
        cream = self.inventory['cream']
        sugar = self.inventory['sugar']
        water = self.inventory['water']
        cup = self.inventory['cup']
        change = self.inventory['change']
        money = self.input_money
    
        # 재료 소진부터 체크
        if (coffee < 10) or (water < 100) or (cup < 0) or (cream < 10) or (sugar < 10):
            print('재료가 부족합니다.')
            print('-'*80)
            print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
            print('-'*80)
            print(f"{money}원을 반환합니다.")
            print('-'*30)
            print("커피 자판기 동작을 종료합니다.")
            print('-'*30)
            stop = True
            return stop
            
        # 커피 판매
        money = money - 300
        coffee = coffee - 10
        water = water - 100
        cup = cup - 1
        change = change + 300
        cream = cream - 15
        sugar = sugar - 10
        print(f"설탕 프림 커피를 선택하셨습니다. 잔액: {money}")
        print('-'*80)
        print(f"재료 현황: coffee: {coffee} cream: {cream} sugar: {sugar} cup: {cup} change: {change}")
        print('-'*80)
        self.update(coffee, cream, sugar, water, cup, change, money)
        stop = False
        return stop
    
    
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
                print('error')
                print('-'*30)
                print("커피 자판기 동작을 종료합니다.")
                print('-'*30)
                break
            
            # 300원 이상인 경우
            elif self.input_money >= 300:
                # 여기도 종료가 안되면 계속 반복
                
                while True:
                    print('-'*30)
                    print(f"  커피 자판기 (잔액:{self.input_money}원)")
                    print('-'*30)
                    
                    self.menu()
                    key = self.input_key()
                    
                    # 또 다른 종료 조건: 돈이 부족
                    if self.input_money < 300:
                        print(f"잔액이 ({self.input_money}원)이 300원보다 작습니다.")
                        print(f"{self.input_money}원이 반환됩니다.")
                        print('-'*30)
                        print('커피 자판기 동작을 종료합니다.')
                        print('-'*30)
                        break
                    
                    # 종료 조건
                    if (key == 5) or (key == None):
                        print(f"종료를 선택하셨습니다. {self.input_money}원이 반환됩니다.")
                        print('-'*30)
                        print('커피 자판기 동작을 종료합니다.')
                        print('-'*30)
                        break
                    
                    # 블랙 커피를 선택
                    elif key == 1:
                        is_stop = self.black_coffee()
                        if is_stop == True:
                            break
                    
                    # 프림 커피를 선택
                    elif key == 2:
                        is_stop = self.cream_coffee()
                        if is_stop == True:
                            break
                    
                    # 설탕 프림 커피를 선택
                    elif key == 3:
                        is_stop = self.sugar_cream_coffee()
                        if is_stop == True:
                            break
                    
                    # 재료 현황을 선택
                    elif key == 4:
                        print('-'*80)
                        print(f"재료 현황: coffee: {self.inventory['coffee']} cream: {self.inventory['cream']} sugar: {self.inventory['water']} cup: {self.inventory['cup']} change: {self.inventory['change']}")
                        print('-'*80)
                    
                # 내부의 while의 반복이 끝났으니 바로 자판기 종료
                break
                
            
            # 300원 보다 작을 경우
            elif self.input_money < 300:
                print(f"투입된 돈 ({self.input_money}원)이 300원 보다 작습니다.")
                print('-'*30)
                print("커피 자판기 동작을 종료합니다.")
                print('-'*30)
                break
            
            
        
        
if __name__ == '__main__':
    # VendingMachine 객체 생성
    inventory_dict = {'coffee':100, 'cream':100, 'sugar':100,
                    'water':500, 'cup':5, 'change':0}
    coffee_machine = VendingMachine(inventory_dict)
    coffee_machine.run()    # VendingMachine 동작 메서드