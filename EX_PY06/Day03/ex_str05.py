# 이스케이프문자 : 특수한 의미를 가지는 문자
# - 형식 : \문자1개
# - '\n' : 줄바꿈 문자
# - '\t' : 탭간격 문자
# - '\'' : 홑따옴표 문자
# - '\"' : 쌍따옴표 문자
# - '\\' : \ 문자, 경로(path), URL관련
# - '\U' : 유니코드
# - '\a' : 알람소리 문자

msg = "오늘은 좋은 날\n내일은 주말이라 행복해"
print(f"msg 줄바꿈 : {msg}")

msg = '오늘은 \'알바가는 날\'이다'
print(msg)

file_path = 'EX_PY06/Day03/test.txt' # 난 이렇게 경로 짧게 쓰는게 좋아.. 내가 일하는 폴더에서 움직이는 거라
print(file_path) # 파일열때는 \오류나던걸로 기억하는데... 문법이 또 바뀐건가?

file_path = 'C:\\Users\\PC\\Desktop\\AI_빅데이터 전문가 양성과정 6기\\EX_PY06\\Day03\\test.txt'
print(file_path)

# r 또는 R : 문자열 내의 이스케이프 문자는 무시됨.
file_path = r'C:\Users\PC\Desktop\AI_빅데이터 전문가 양성과정 6기\EX_PY06\Day03\test.txt'
print(file_path)

