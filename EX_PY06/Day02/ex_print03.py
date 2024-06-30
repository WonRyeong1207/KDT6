# 내장함수 print 사용법3
# - flie : 데이터를 파일에 기록, 불러오기

# 파일 읽기 & 쓰기
file_name = 'test.txt'
file_path = 'EX_PY06/Day02/test.txt'
f = open(file_path, mode='w', encoding='utf-8') # 파일을 쓰기 모드로 열기
print("雨", file=f) # 문자 깨진다...
print("ame", file=f)
f.close()

