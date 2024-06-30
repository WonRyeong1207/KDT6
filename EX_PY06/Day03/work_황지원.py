# 10장부터 11장까지

# p.109 연습문제
a = list(range(5, -10, -2))
print(a)

# p.110 심사문제
n = int(input())
# n = 5
t = tuple(range(-10, 10, n))
print(t)

# p.140 연습문제 1
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
population = [10249679, 10195318, 10143635, 10103233,
              10022181, 9930616, 9857426, 9838892]

print(year[-3:])
print(population[-3:])

# p.141 연습문제 2
n = -32, 75, 97, -10, 9, 32, 4, -15, 0, 76, 14, 2
print(n[1::2])

# p.141 ~ 142 심사문제 1
x = input().split()
# x = ['oven', 'bat', 'pony', 'total', 'leak', 'wreck', 'curl']
del x[-5]
x = tuple(x)
print(x)

# p.142 심사문제 2
s1 = input()
s2 = input()
# s1 = 'python'
# s2 = 'python'
print(s1[1::2]+s2[::2])
