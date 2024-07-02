# 반복문

# 실습 : 출력하고 싶은 단을 입력 받아서 해당 단의 구구단을 출력

n = int(input("단 : "))

# for i in range(1, 10):
#     print(f"{n} * {i} = {n*i}")
# print()

print("{0:-^40}".format(f"  {n} 단  "))
for i in range(1, 10, 3):
    print("{0: <10}".format(f"{n} * {i} = {n*i}"), end='    ')
    print("{0: <10}".format(f"{n} * {i+1} = {n*(i+1)}"), end='    ')
    print("{0: <10}".format(f"{n} * {i+2} = {n*(i+2)}"))

