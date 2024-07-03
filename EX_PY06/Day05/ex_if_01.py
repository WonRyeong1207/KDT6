# 조건문 표현식
# - 조건문을 1줄로 축약해주는 문법
# - 다중 조건문을 축약할 때 사용
# - 다른 프로그래밍 언어에서 삼항연산자와 유사
# - 형식 : 참 실행코드 if 조건식 else 거짓 실행코드

# 실습 : 임의의 숫자 데이터 정하기
# - 짝수인지 홀수인지 판별

num = int(input("수를  입력하세요 : "))

if (num == 0):
    print("입력하신 숫자는 0입니다.")
elif ((num%2) == 0 ): # if num%2: 값이 있으면 True 홀수 / if not num%2: 결과는 짝수
    print(f"입력하신 숫자 {num}는 짝수입니다.")
else:
    print(f"입력하신 숫자 {num}는 홀수입니다.")
    
# 1줄로 조건식을 축약
print("짝수") if (num%2 == 0) else print("홀수")
print("홀수") if num%2 else print("짝수")
print("짝수") if not num%2 else print("홀수")

