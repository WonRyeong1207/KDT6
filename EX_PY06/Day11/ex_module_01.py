# 파이썬 모듈

# 파이썬 파일 한개
# - 구성 : 변수, 함수, 클래스가 존재
#        반드시 모두 다 있지는 않음.
# - 종류 : 내장모듈 / 사용자정의모듈 / 써드파티션모듈(설치필수)

# 사용법 => 현재 파이썬 파일에 포함 시켜야 사용가능
import math # 확장자는 적지 않음
import random
from matplotlib import pyplot as plt # 모듈의 이름이 길어서 별칭을 줌
import matplotlib.pyplot as plt # 이렇게도 쓸 수 있음
# 별칭을 지정했을 시에는 별칭으로만 사용해야함.

# 모듈 내의 변수, 함수, 클래스 사용 방법
# 모듈명.변수명 , 모듈명.함수명(), 모듈명.클래스명()
print(f"내장모듈 math 안에 있는 pi 변수 : {math.pi}")

print(f"내장모듈 math 안에 있는 factorial함수 3! : {math.factorial(4)}")

print(f"내장모듈 random 안에 있는 random() 함수 10 : {random.Random(10)}")
