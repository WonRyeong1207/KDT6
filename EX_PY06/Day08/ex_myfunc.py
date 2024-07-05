# 내가 만든 함수들

def add(num1, num2): return num1 + num2

# 합수 기능: 입력 데이터가 유효한 데이터인지 검사해주는 기능
# 함수 이름 : check_data
# 매개 변수 : 문자열 데이터, 데이터 개수 data, cnt, sep=' '
# 함수 결과 : 유효 여부 T/F

def check_data(data, cnt, sep=' '):
    if len(data):
        data_list = data.split(sep)
        if cnt == len(data_list):
            return True
        else:
            False
    else:
        return False




# system 에서 만들어진 매직함수
# 파이썬은 다른 파일에서 불러올 수 있으니까.
# 필요한 것들 외에 다른 것도 다 불려가버려서
# import 되면 __name__에 파일명을 넣어줌.

# print(f"__name__:{__name__}")

# 불러들어온 파일에 나옴
# 안나오게
if __name__ == '__main__':
    print("--test--")
    print(f"result : {add(100,100)}")
    