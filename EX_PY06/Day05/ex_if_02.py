# 1줄로 축약 : 조건부표현식

# 실습 : 문자 1개 코드값을 저장하는 조건식을 저장
# - 알파벳(a-z, A-Z) 코드값으로 반환, 그외는 None으로 코드값 반환

data = 'N'
# data = input()

if (('a' <= data <= 'z') or ('A' <= data <= 'Z')):
    print(f"입력한 문자 {data}의 코드값 : {ord(data)}")
else:
    print(None)
    
result = ord(data) if (('a' <= data <= 'z') or ('A' <= data <= 'Z')) else None
print(f"결과 : {result}")


