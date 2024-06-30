# 연산자
# 3. 논리 연산자
#   - 종류 : and, or, not
#   - 특징 : 여러개의 경우에 대한 결과를 바탕으로 결론을 내림
#   - and : 논리곱 1*0 = 0, 1*1 = 1
#   - or : 논리합 1+0 = 1, 1+1= 1
#   - not : 부정 결과를 뒤집음. 인버터

num1 = 10
num2 = 7

print(f"{num1}>0 and {num2}>0 : {(num1>0) and (num2>0)}")
print(f"{num1}>0 or {num2}==0 : {(num1)>0 or {num2==0}}")
print(f"not {num1}<0 : {not (num1<0)}")
