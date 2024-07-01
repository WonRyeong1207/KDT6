"""
price = int(input())
coopon = input()

if (coopon == 'Cash3000'):
    price = price - 3000
if (coopon == 'Cash5000'):
    price = price - 5000
    
print(price)
"""

'''
k, e, m, s = map(int, input().split())

if ((k<0) or (k>100) or (e<0) or (e>100) or (m<0) or (m>100) or (s<0) or (s>100)):
    print("잘못된 점수")
    
if (((k+e+m+s)/4)>=80):
    print("합격")
else:
    print("불합격")
'''

'''
n = int(input("수 입력 : "))

for i in range(n):
    for j in range(n+1,i,-1):
        if (j>=i):
            print(' ', end='')
    if (i == 0):
        print('*',end='')
    else:
        print('*'*(i+(i*2)),end='')
    print()
'''

'''
import turtle as t

t.penup()
t.lt(90)
t.fd(150)
t.rt(90)
t.pendown()
color = ['red', 'orange', 'yellow', 'green', 'skyblue', 'blue', 'indigo', 'puple']

n = 7
for i in range(n):
    t.color(color[i])
    t.fd(100)
    t.rt((360/n)*2)
    t.fd(100)
    t.lt(360/n)
t.mainloop()
'''

# 지뢰찾기
'''
row, col = map(int, input("matrix size : ").split())
matrix = []
for i in range(row):
    matrix.append(list(input("field state : ")))
    
print(matrix)
print()
'''
row, col = 3, 3
matrix = [['.', '*', '*'], ['*', '.', '.'], ['.', '*', '.']]

for i in range(row):
    current_field = []
    fild_field = []
    num = 0
    for j in range(col):
        if (matrix[i][j] == '*'):
            current_field.append('*')
        if (matrix[i][j] == '.'):
            # 내주변 위아래좌우, 대각선도 봐야하네...
            num = 0 # 중복해서 더해지는 것을 방지하기 위해서
            
            # i-1
            if (((i-1) >= 0) and((j-1) >= 0) and (matrix[i-1][j-1] == '*')):
                num = num + 1
            if (((i-1) >= 0) and (matrix[i-1][j] == '*')):
                num = num + 1
            if(((i-1) >= 0) and ((j+1) < col) and (matrix[i-1][j+1] == '*')):
                num = num + 1
            
            # i
            if (((j-1) >= 0) and (matrix[i][j-1] == '*')):
                num = num + 1
            if (((j+1) < col) and (matrix[i][j+1] == '*')):
                num = num + 1
            
            # i +1
            if (((i+1) < row) and ((j-1) >= 0) and (matrix[i+1][j-1]) == 0):
                num = num + 1
            if (((i+1) < row) and (matrix[i+1][j] == '*')):
                num = num + 1
            if (((i+1) < row) and ((j+1) < col) and (matrix[i+1][j+1])):
                num = num + 1
                
            current_field.append(num)

        # current_field = ''.join(current_field)
    print(current_field)
    # fild_field.append(current_field)
    
# for i in range(row):
#    print(fild_field[i])