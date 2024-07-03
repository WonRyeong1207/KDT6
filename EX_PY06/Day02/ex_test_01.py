# 입력 & 출력 실습
# 실습 1 : 데이터 저장 및 변수 생성 그리고 출력
# - 생년월일
# - 띠 , 혈액형

year, month, day = '2001', '01', '01'
chinese_zodiac = '용'
blood_type = 'B'
blood_feature = {'A':'수줍은', 'B':'만사 귀찮은', 'O':'대담한', 'AB':'엉뚱한'}

print(f"나는 {year}년 {month}월 {day}일 {chinese_zodiac}띠입니다.")
print(f"나는 {blood_feature[blood_type]} {blood_type}형입니다.")

# 실습 2 : 입력받은 데이터를 저장. 파일로 저장할 것
# - 좋아하는 계절, 나라, 여행가고 싶은 나라

season, country, want_country = input("좋아하는 계절, 좋아하는 나라, 여행가고 싶은 나라 : ").split(',')

flie_path = 'EX_PY06/Day02/perfer.txt'
f = open(flie_path, mode='w', encoding='utf-8') # 한글이 다깨져서 저장되는 경우
f.write(f"좋아하는 계절 : {season} \n")
f.write(f"좋아하는 나라 : {country} \n")
f.write(f"여행가고 싶은 나라 : {want_country}")
f.close()