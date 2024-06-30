# 내장함수

# 숫자 데이터 절대값 계산해주는 : 함수 abs()

print(abs(-9))

# 최댓값, 최솟값 찾아주는 함수 : max(), min()

print(max(10,3), min(10,3))
num_list = [10, 20, 30, 40]
print(f"max : {max(num_list)}, min : {min(num_list)}")

# 제곱근 계산 내장함수

print(f"연산자 2**4 : {2**4}")
print(f"내장함수 pow() : {pow(2,4)}")

# 파일관련 내장함수
# open(파일명, 동작모드(기본 읽기모드), 인코딩(기본이 시스템을 따라감))

FILE_PATH = 'EX_PY06/Day03/word.txt' # 파일 경로지정
f = open(FILE_PATH, mode='w', encoding='utf-8')
f.write("hello \n")
f.write("おはいよ \n")
f.write("雨はきらい。\n")
f.write("민원시스템 너무 느려...")
f.close()