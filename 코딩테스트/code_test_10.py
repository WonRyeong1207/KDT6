# 10번. 팩토리얼(Factorial)을 while 반복문으로 구현해 주세요.
#       팩토리얼 수를 입력 받아서 팩토리얼 결과를 아래와 같이 출력하세요.
# 예시
# 입력 : 팩토리얼 수 입력 : 7
# 출력 : 7! => 7 * 6 * 5 * 4 * 3 * 2 * 1 = 5040

num = int(input("팩토리얼 수 입력 : "))

def Factorial(n, steps):
    if n <= 1:
        return 1
    steps.append(n)
    return n * Factorial(n-1, steps)

step = []
factorial = Factorial(num, step)
step.append(1)
step_str = " * ".join(map(str, step))
print(f"출력 : {num}! => {step_str} = {factorial}")