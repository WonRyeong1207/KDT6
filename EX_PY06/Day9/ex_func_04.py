# 함수 funtion 이해 및 활용
# 함수 기반 계산기 프로그램
# - 4칙 연산 기능별 함수 생성 => 덧셈, 뺄셈, 곱셈, 나눗셈
# - 2개의 정수만 계산

# 4칙 연산 함수
def add(n1, n2):
    result = n1 + n2
    return result

def minus(n1, n2):
    result = n1 - n2
    return result

def multi(n1, n2):
    result = n1 * n2
    return result

def div(n1, n2):
    if n2 == 0:
        result = 'None'
    else:
        result = n1 / n2
    return result

# 메뉴를 출력하는 함수
# 함수 기능 : 계산기 메유를 출력하는 함수
# 함수 이름 : menu
# 매개 변수 : None
# 함수 결과 : None

def menu():
    print(f"{'*':*^16}") # > : 오른쪽 정렬, < : 왼쪽 정렬 , ^ : 가운데 정렬
    print(f"*{'계산기':^11}*")
    print(f"{'*':*^16}")
    print(f"{'* 1. 더하기'}{'*':>5}")
    print(f"{'* 2. 뻬기'}{'*':>7}")
    print(f"{'* 3. 곱하기'}{'*':>5}")
    print(f"{'* 4. 나누기'}{'*':>5}")
    print(f"{'* 5. 종료'}{'*':>7}")
    print(f"{'*':*^16}")

# menu()

# 함수 기능: 연산 수행후 결과를 반환하는 함수
# 함수 이름 : calc
# 매개 변수 : 함수명, 숫자 str 2개
# 함수 결과 : 연산 결과, print

def calc(func, data):
    if check_data(data, 2):
        num1, num2 = map(int, data.split())
        print(f"결과: {func(num1, num2)}\n")

# 함수 기능 : 입력받은 데이터가 유효한 데이터인지 검사하는 함수
# 함수 이름 : check_data
# 매개 변수 : str 데이터 , 데이터 개수
# 함수 결과 : True or False

def check_data(data, sp_num):
    data = data.split()
    if len(data) == sp_num:
        isOK = True
        for d in data:
            if not d.isdecimal():
                isOK = False
                print("0~9 사이의 정수만 입력하세요.\n")
                break
            return isOK
    else:
        print("개수가 맞지 않습니다.\n")
        return False

# 계산기 프로그램
# - 사용자에게 원하는 계산을 선택하는 메뉴를 출력
# - 1: +, 2: -, 3: *, 4: /, 5: 종료

# 무한반복 # op 연산자를 추가로 입력 받을 수 있음. 그러면 전체 다 수정해야함.
while True:
    menu()
    
    # 메뉴 선택 요청
    # choice = int(input("메뉴 선택: ")) # 제대로 입력했다는 전제 조건
    choice = input("메뉴 선택: ")
    if choice.isdecimal():
        choice = int(choice)
    else:
        print("0~9 사이의 정수만 입력하세요.\n")
        continue
    
    # 종료 조건
    if choice == 5:
        print("프로그램을 종료합니다.")
        break
    
    elif choice == 1:
        print('더하기')
        nums = input("정수 2개(예: 10 2): ")
        calc(add, nums)
        
    elif choice == 2:
        print('빼기')
        nums = input("정수 2개(예: 10 2): ")
        calc(minus, nums)
        
    elif choice == 3:
        print('곱하기')
        nums = input("정수2개(예: 10 2): ")
        calc(multi, nums)
        
    elif choice == 4:
        print("나누기")
        nums = input("정수2개(예:10 2): ")
        calc(div, nums)
        
    else:
        print("선택된 메뉴가 없습니다.\n")