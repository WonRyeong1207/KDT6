# 9번. 월(Month)을 입력 받아 해당 월(Month)의 영어와 계절을 출력하는 코드를 작성하세요.
# 조건
# - 월(Month)에 해당하지 않는 숫자 입력 시 "잘못된 데이터 입니다" 출력
# 예시
# 입력 : 좋아하는 월 입력 : 3 / 출력 : March Spring

month = int(input("좋아하는 월 입력 : "))
season = {1:'January Winter', 2:'February Winter', 3:'March Spring',
          4:'April Spring', 5:'May Spring', 6:'June Summer',
          7:'July Summer', 8:'August Summer', 9:'September Autumn',
          10:'October Autumn', 11:'November Autumn', 12:'December Winter'}


if ((month < 1) or (month > 12)):
    print("잘못된 데이터 입니다")
    month = int(input("좋아하는 월 입력 : "))
    
print(season[month])