# 1번. 문자열 리스트를 입력 받아서 내림차순 결과 가장 낮은 문자열과 높은 문자열을 출력하는 코드를 구현하세요.
# 입력 : msg = ['Good', 'child', 'Zoo', 'apple', 'Flower', 'zero']
# 출력 : 정렬 결과 : ['zero', 'child', 'apple', 'Zoo', 'Good', 'Flower']
#        가장 높은 문자열 : zero, 가장 낮은 문자열 : Flower

msg = input("문자열을 입력하세요.(공백으로 분리합니다.) : ").split()

msg.sort(reverse=True)
high_list = msg[0]
low_list = msg[-1]

print(f"정렬 결과 : {msg}")
print(f"가장 높은 문자열 : {high_list}, 가장 낮은 문자열 : {low_list}")
