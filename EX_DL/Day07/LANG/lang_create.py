# 모듈로딩 ---------------------------------------
import glob, os.path, re, json

# 데이터 변수 선언 --------------------------------
DEBUG = True
PATH= '../../data/lang/'
FILENAME=PATH+'freq_train.json'
FILENAME2=PATH+'freq_test.json'

# 언어데이터 파일 생성 ----------------------------------------
if not os.path.exists(PATH): os.mkdir(PATH)

# 텍스트 중 알파벳 출현 빈도 조사 함수 --------------------------------
def check_freq(fname):
    # 파일명만 추출
    name = os.path.basename(fname)
    if DEBUG: print('name => {}'.format(name))

    # 파일이름 앞에 알파벳 2개 시작 파일 체크
    # 예) en-1.txt
    lang = re.match(r'^[a-z]{2,}', name).group()   # 알파벳으로 시작해서 2개 이상, 일치하는 문자열 리턴
    if DEBUG: print(lang)

    with open(fname, mode="r", encoding="utf-8") as f:
        text = f.read()

    # txt파일의 문자 검사 시작
    text = text.lower()   # 문자열 소문자료 변환

    # 숫자 세기 변수(cnt) 초기화
    cnt = [0 for n in range(0, 26)]         # 알파벳 26개 리스트 생성 및 0으로 초기화
    if DEBUG: print('len(cnt) {}, cnt {} '.format(len(cnt), cnt))

    # 문자의 코드값 읽기
    code_a = ord("a")
    code_z = ord("z")
    # 알파벳 출현 횟수 계산
    for ch in text:
        n = ord(ch)                     # 문자의 코드값 변환
        if code_a <= n <= code_z:       # a~z 사이에 있을 때
            cnt[n - code_a] += 1

    # 모든 검사 후 a~z 출현갯수 확인
    if DEBUG: print('cnt {}, len(cnt) {}'.format(cnt, len(cnt)))

    # 출현빈도로 변환
    total = sum(cnt)                            # txt파일에 출력된 a~z까지 전체 갯수
    freq = list(map(lambda n: n / total, cnt))
    if DEBUG: print('len(freq) {},  freq{}'.format(len(freq),freq))
    return (freq, lang)
    
# 파일 로딩 기능 ------------------------------------------------------
def load_files(path):
    freqs = []
    labels = []
    file_list = glob.glob(path)     #확장자 txt인 모든 파일리스트
    if DEBUG: print('load_files() => file_list = {}'.format(file_list))

    # 파일별로 a-z까지 빈도 , 언어코드라벨
    for fname in file_list:
        r = check_freq(fname)   # 파일별 알파벳 출현빈도 체크
        if DEBUG: print('r => {}, {}, {}'.format(len(r[0]), len(r[1]), r))

        freqs.append(r[0])      # a~z까지 출현빈도 데이터 추가
        labels.append(r[1])     # 언어 라벨 추가

    if DEBUG:
        print('freqs = {}, {}'.format(freqs, len(freqs)))
        print('labels = {}, {}'.format(labels, len(labels)))
    return {"freqs":freqs, "labels":labels}

# (1) 데이터 준비하기 ----------------------------------------
data = load_files(PATH+"train/*.txt")  # 결과 dictionary 타입 리턴
test = load_files(PATH+"test/*.txt")   # 결과 dictionary 타입 리턴

# JSON으로 결과 저장하기
with open(FILENAME, "w", encoding="utf-8") as fp:
    json.dump(data, fp)     # json 문자열 데이터로 변환

with open(FILENAME2, "w", encoding="utf-8") as fp:
    json.dump(test, fp)     # json 문자열 데이터로 변환

if DEBUG: print("test =>", test)
