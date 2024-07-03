# 반복문과 continue
# - continue 구문을 만나면 구문 아래 코드 실행X
# - 반복문으로 가서 다음 요소 데이터를 가지고 진행

# 실습 : 1~50까지 숫자 데이터
# 3의 배수인 경우에만 화면에 출력

for i in range(1, 50+1):
    if ((i%3) == 0):
        print(i, end=' ')
print()

for i in range(1, 50+1):
    if (i%3):
        continue
    else:
        print(i, end=' ')      
print()
