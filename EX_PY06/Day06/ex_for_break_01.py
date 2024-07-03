# 제어문 - 반복문 중단 break
# - 반복을 중단 시키는 조건문과 함께 사용됨.

# 실슴 : 숫자 데이터의 합계가 30 이상이 되면 합계 멈춤.
# 숫자 데이터는 1~50으로 구성됨.

num = [x for x in range(1, 51)]
t_n = 0

for i in num:
    if (t_n >= 30):
        t_n = t_n - i
        break # 즉시 반복 종료
    else:
        t_n = t_n + i
print(t_n)
print()

# 실습 : 4개의 과목점수가 있음.
# - 한 과목이라도 점수가 40점 이하면 불합격, 4개 과목 평균이 60점 이상이면 합격

score = [42, 90, 90, 60]
mean = 0

for i in score:
    if (i <= 40):
        print('과락입니다')
        break
    
    # mean = sum(score) / len(score)
    mean = (mean + i) /len(score)        
    if (mean < 60): # 점수를 알고 ㅅㅣㅍ지 않아서
        print('불합격입니다')
        break
    else:
        print('합격입니다.')
print()
