# 리스트 내부 요소 변경

datas = list(range(1, 12, 2))
print(f"datas : {datas}")

# 데이터 변경
datas[0] = 100
print(f"datas[0] : {datas[0]}")

# 데이터 삭제
del datas[0]
print(f"datas : {datas}")
print(f"datas[0] : {datas[0]}")

datas[:3] = ['삼', '오', '칠']
print(f"datas : {datas}")

datas[:3] = list(range(11,99,11))
print(f"datas : {datas}")

del datas[8]
del datas[:4]
print(f"datas : {datas}")
