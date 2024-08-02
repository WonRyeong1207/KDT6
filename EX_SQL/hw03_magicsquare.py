# kdt6_황지원
# 화면상에서 홀수를 입력받고 해당하는 n x n 형태의 마방진을 구형하시오.

# 구현기능
# - 홀수차 배열의 크기 입력: 짝수 입력 시 오류 처리 및 다시 입력
# - 입력된 배열의 크기에 따라 n x n 크기의 이차원 배열 생성 (3, 5, 7, 9 확인)
# - 정상적인 홀수차 마방진 구현 및 화면 출력
#   - 출력시 자리 수 맞춤
#   - 오류 발생시 각 -2정 감점
#   - 반복은 필요 없음 (1회 실행 후 종료)

# 마방진의 원리
# - 마방진 : 행과 열이 같은 정방 행렬
# - 일반적으로 n x n 마방진에서 각 줄의 합은 n(n^2+1)/2
# - 가로, 새로, 대각선의 합이 항상 같음

# 마방진의 원리 (3, 3) 기준
# - 시작위치 : 첫 행의 가운데 열에서 시작함 (1, 0) 3//2 - 가운데 나눗셈 몫
# - 이동 규칙
#   - 다음 위치는 오른쪽 대각선 방향으로 이동
#   - y축 방향으로 범위가 벗어난 경우, y는 마지막 행으로 이동
#   - x축 방향으로 범위가 벗어난 경우, x축 첫 번째 행으로 이동
#   - 다음 이동 위치에 이미 값이 있는 경우, y는 y + 1

# 입력을 받고 print 하는 함수
def select_num():
    while True:
        num = input("홀수차 배열의 크기를 입력하세요: ")
        
        if num.isdecimal() == True:
            num = int(num)
            if (num%2) == 0:
                print('짝수를 입력하였습니다. 다시 입력하세요')
                continue
            else:
                if (num == 3) or (num == 5) or (num == 7) or (num == 9):
                    print(f"Magic Square ({num} x {num})")
                    break
                else:
                    print("3, 5, 7, 9 중 하나를 입력하세요.")
                    continue
        else:
            print('정수를 입력해주세요')
            continue
    return num

# test
# num = select_num()

# 마방진을 만들어가는? 함수?
def create_map(num):
    # 필요한 초깃값 설정
    map_array = [[0 for i in range(num)] for i in range(num)]
    num_list = [n for n in range(1, num*num + 1)]
    x, y, n = num//2, 0, 0  # x=0, y =1

    while True:
        # 종료조건
        if n == (num*num):
            break
        
        # 여기에 둘다 범위를 벗어나는 경우
        if (y < 0) and (x >= num):
            y = y + 2
            x = x - 1
            if (map_array[y][x] != 0):
                pass
                
            else:
                map_array[y][x] = num_list[n]
                n = n + 1
                x = x + 1
                y = y - 1
        
        # y 범위만 벗어난 경우
        if y < 0:
            y = num -1

            if x >= num:
                x = 0
                if (map_array[y][x] != 0):
                    y = y - 1
                    x = x + 1
                    
                else:
                    map_array[y][x] = num_list[n]
                    n = n + 1
                    x = x + 1
                    y = y - 1
                    
            else:
                map_array[y][x] = num_list[n]
                n = n + 1
                x = x + 1
                y = y - 1
                
        # x만 범위를 벗어난느 경우
        elif x >= num:
            x = 0
            if (map_array[y][x] != 0):
                y = y + 2
                x = x - 1
                
            else:
                map_array[y][x] = num_list[n]
                n = n + 1
                x = x + 1
                y = y - 1
        
        # 범위를 벗어 나지 않았을때
        else:
            if (map_array[y][x] != 0):
                y = y + 2
                x = x - 1
                
            else:
                map_array[y][x] = num_list[n]
                n = n + 1
                x = x + 1 
                y = y - 1
                
            
    return map_array

# 만든 마방진을 print하는 함수
def print_magicsquare(map_array, num):
    for row in range(num):
        for col in range(num):
            print(f"{map_array[row][col]:2}", end=' ')
        print()

if __name__ == '__main__':
    num = select_num()
    print_magicsquare(create_map(num), num)
