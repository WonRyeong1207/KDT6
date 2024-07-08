# 함수명에 대하여
#  - 코드가 있는 부분에 붙여진 이름
#  - 코드가 시작되는 주소를 저장하고 있음

# 내장함수
show = print # 함수의 주소를 저장했기 때문에
show("happy") # 같은 기능을 사용할 수 있음. ex) 클래스에서 객체 설정할때

datas = [11, 22, 33]
func = [max, min, sum] # 이렇게 자주쓰는 함수는 이름을 가지고 와서 사용할 수 있음.
print(func[0](datas), func[1](datas), func[2](datas))

