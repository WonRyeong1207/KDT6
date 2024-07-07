# P.285~286 심사문제 - 지뢰찾기

# 행렬의 크기와 map_state를 받음
row, col = map(int, input("matrix size : ").split())
matrix = []
for i in range(row):
    matrix.append(list(input("field state : ")))
print() # 결과코드를 편하게 구분해서 보기 위함.

fild_field = [] # 최종 결과를 보여줄 리스트

for i in range(row):
    current_field = [] # 현재 진행 상황을 보기 위한 리스트, 행이 바뀔때 마다 초기화
    
    for j in range(col):
        if (matrix[i][j] == '*'):
            current_field.append('*')
            
        if (matrix[i][j] == '.'):
            # 내주변 위아래좌우, 대각선도 봐야하네...
            num = 0 # 중복해서 더해지는 것을 방지하기 위해서 열이 바뀔때마다 초기화
            
            # 리스트를 사용하기 때문에 i나 j값이 0보다 작아지면 뒤에서부터 넘버링됨.
            # 이를 방지하기 위해 0보다 크거나 같은 경우를 비교함.
            # 또한 입력받은 row나 col보다 크면 인덱스 에러가 나기 때문에 그보다 작은 범위에서만 움직임.
            # if문의 분기를 if, elif를 안 하는 이유는 elif 사용시 지뢰를 하나만 탐색함.
            # 여러개의 지뢰를 찾아야하기에 if로만 사용함.
            
            # i-1
            # [i-1][j-1]
            if (((i-1) >= 0) and((j-1) >= 0) and (matrix[i-1][j-1] == '*')):
                num = num + 1
            # [i-1][j]
            if (((i-1) >= 0) and (matrix[i-1][j] == '*')):
                num = num + 1
            # [i-1][j+1]
            if (((i-1) >= 0) and ((j+1) < col) and (matrix[i-1][j+1] == '*')):
                num = num + 1
            
            # i
            # [i][j-1]
            if (((j-1) >= 0) and (matrix[i][j-1] == '*')):
                num = num + 1
            # [i][j+1]
            if (((j+1) < col) and (matrix[i][j+1] == '*')):
                num = num + 1
            
            # i +1
            # [i+1][j-1]
            if (((i+1) < row) and ((j-1) >= 0) and (matrix[i+1][j-1] == '*')):
                num = num + 1
            # [i+1][j]
            if (((i+1) < row) and (matrix[i+1][j] == '*')):
                num = num + 1
            # [i+1][j+1]
            if (((i+1) < row) and ((j+1) < col) and (matrix[i+1][j+1] == '*')):
                num = num + 1
            
            num = str(num) # 마지막에 str로 출력하기 위해서는 리스트 내부 요소가 str 형태를 취해야함.
            current_field.append(num)

    # print(current_field)
    fild_field.append(current_field) # 최종결과만 담음

for i in range(row):
    print(''.join(fild_field[i])) # str 형태로 출력하기 위해서 사용

# .는 0, *는 1에 padding=1, stride=1, 필터는 3x3(전부 1)로 converlution을 한다면 계산하기는 더 쉬웠을 것 같아 보임.
# 어떻게 할지는 고민....