# 실습 : 글자를 입력 받습니다.
#       - 입력 받은 글자 (a~Z, A~Z) 코드값을 출력합니다.

a = input("문자를 하나 입력하세요 (a ~ z, A ~ Z): ")
# print(f"문자 {a}의 코드값은 {ord(a)} 입니다.")
if ((len(a) == 1) and ('a' <= a <='z') or ('A' <= a <= 'Z')):
    print(f"입력한 문자 {a}의 코드값은 {ord(a)} 입니다.")
else:
    print("입력하신 문자가 없거나 형식에 맞지 않습니다.")

print()

# 여러개
# data 'ab'
# print(list(map(ord, data)))
# len()이 0이 나오면 거짓이니까 if (len(a) and xxx): 으로 처리해도 됨.

# 실습 : 점수를 입력 받은 후 학점을 출력 합니다.
# - 학점 : A+(95초과), A(95), A-(90이상 95미만), B+, B, B-, C+, C, C-, D+, D, D-, F

score = int(input("점수를 입력하세요 : "))
score_rank = ''

if ((score < 0) or (score > 110)):
    print("잘못된 점수를 입력하셨습니다.\n다시 입력해주세요.")
    score = int(input("점수를 입력하세요 : "))

if (95 < score):
    score_rank = 'A+'
elif (score == 95):
    score_rank = 'A'
elif (90 <= score < 95): # 전부 if가 아니니까 조건은 하나만 있어도 됨. elif (90 <= score): xxx
    score_rank ='A-'
    
elif (85 < score < 90): # 여기도 elif (85 < score): xxxx
    score_rank = 'B+'
elif (score == 85):
    score_rank = 'B'
elif (80 <= score <85):
    score_rank = 'B-'
    
elif (75 < score < 80):
    score_rank = 'C+'
elif (score == 75):
    score_rank = 'C'
elif (60 <= score < 75):
    score_rank = 'C-'
    
elif (55 < score < 60):
    score_rank = 'D+'
elif (score == 55):
    score_rank = 'D'
elif (50 <= score < 55):
    score_rank = 'D-'
else:
    score_rank = 'F'
    
print(f"학점은 {score_rank} 입니다.")

