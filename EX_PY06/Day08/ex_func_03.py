# 사용자 정의 함수 실습

# 함수 기능 : 원하는 단의 구구단을 출력하는 기능 함수
# 힘수 이름 : gugu
# 매개 변수 : dan
# 함수 결과 : 2단이면 2단의 출력결과를 보여줌

def gugu(dan):
    for i in range(10):
        if (i == 0):
            print(f"-- {dan} 단 --")
        else:
            print(f"{dan} * {i} = {dan*i:>2}")

n = int(input("출력을 원하는 단 : "))
gugu(n)
print()

# 함수 기능 : 파일의 확장자를 반환해주는 기능 함수
# 함수 이름 : find_extan
# 매개 변수 : list(data)
# 함수 결과 : '.xxx'

def find_extan(data):
    carry = []
    for i in data:
        idx = i.rfind('.')
        carry.append(i[idx:])
    return carry
    
files = list(map(str, input("파일명을 입력 (예 xxx.jpg xxxx.hwp) : ").split()))
print(find_extan(files))
print()
