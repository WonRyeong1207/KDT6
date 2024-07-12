# 클래스(class)
# - 객제지향언어(OOP)에서 데이터를 정의하는 자료형
# - 데이터를 정의할 수 있는 데이터가 가진 속성과 기능 명시
# - 구성요소 :  속성/attribute/field + 기능/method

# 클래스 정의 : 햄버거를 나타내는 클래스
# 클래스 이름 : bugger
# 클래스 속성 : 번, 패티, 야채, 치즈
# 클래스 기능 : 햄버거 설명

class Bugger:
    # 공통으로 쓰는건 빼버림
    kind = '맥도날드'
    
    # 클래스 내부 객체 초기화
    def __init__(self, bread, patty, veg):
        self.bread = bread
        self.patty = patty
        self.veg = veg
        # self.kind = kind
    
    # 기능, 메서드
    def print_info(self):
        print(f"브랜드 : {self.kind}")
        print(f"빵 종류 : {self.bread}")
        print(f"패 티 : {self.patty}")
        print(f"야 채 : {self.veg}")
        
        
# 클래스 사용하가
# 객체를 만들고
bugger1 = Bugger('브리오슈', '불고기', '양상추, 토마토, 양파')
# 객체의 메소드 사용
bugger1.print_info()
print()

bugger2 = Bugger('참깨곡물빵', '쇠고기패티', '치즈, 양상추, 양파, 토마토')
bugger2.print_info()