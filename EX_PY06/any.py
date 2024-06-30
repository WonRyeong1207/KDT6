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

