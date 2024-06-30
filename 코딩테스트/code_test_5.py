# 5번. 문자열을 입력하면 코드값을 아래와 같이 출력해주는 함수를 구현해 주세요.
# 입력 : data = "가나다"
# 출력 : "가나다"의 인코딩 : 0xac000xb0980xb2e4
#        "가나다" 인코딩 : 0b1010110000000000b10110000100110000b1011001011100100

data = input("문자열을 입력해 주세요 : ")

def EnCodeData(str_data):
    
    char_en_hex = []
    char_en_bin = []
    for i in range(len(str_data)):
        en_char = str_data[i].encode()
        char_en_hex.append(f'0x{en_char.hex()}')
        char_en_bin.append(f'0b{"".join(format(byte, "08b") for byte in en_char)}')
        

    hex_encoding = ''.join(char_en_hex)
    bin_encoding = ''.join(char_en_bin)
        
    
    print(f'"{str_data}"의 인코딩 : {hex_encoding}')
    print(f'"{str_data}" 인코딩 : {bin_encoding}')
    
EnCodeData(data)