# 내장함수

# 정수관령 내장함수
# 2진수(컴퓨터), 8진수, 10진수(사람), 16진수(프로그래밍, 사람시점)

# 2진수 : bin() ==> 0b
print(4, bin(4), type(bin(4)))

# 8진수 : oct() ==> 0o
print(8, oct(8), type(oct(8)))

# 10진수 : int() 제공해주는 것이 없음... 만들어야함.. int로 사용가능
print(4, int(4))

# 16진수 : hex() ==> 0x
print(20, hex(20), type(hex(20)))

# 16진수 ==> 10진수 변환해주는 내장함수 int()
print('0x14', int('0x14', base=16))
# 기본은 10진수이기에 base로 형식을 지정해줘야 함. 위는 str일때
print(0x14, int(0x14))
# ''없이 수를 입력하면 10진수 형식으로 출력됨.
