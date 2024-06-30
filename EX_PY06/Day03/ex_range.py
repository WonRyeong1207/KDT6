# range()
# - 숫자 범위를 생성하는 내장함수
# - 형식 : range(start, end+1, jump)

# 1 ~ 100 숫자 저장하세요.
num_list = [x for x in range(1, 101)]
print(num_list, '\n', type(num_list), '\n', len(num_list))

print(num_list[0], num_list[-1])
print(num_list[30:40], num_list[::5])

# 실습1 1~100에서 3의 배수만 저장
three_num_list = [x for x in range(1, 101) if ((x%3)==0)]
three_num_list2 = list(range(3, 101, 3))
print(three_num_list)
print(three_num_list2)
print('\n\n')

# 실습2 1.0에서 10.0까지 저장
num_flo_list = [float(x) for x in range(11)]
print(num_flo_list)
