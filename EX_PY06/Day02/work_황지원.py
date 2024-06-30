# 3장부터 9장까지
# p.49 연습문제
print('Hello, world!')
print('Python Programming')
print('\n\n')

# p.49 심사문제
print("Hello, world!")
print("Hello, world!")
print('\n\n')

# p.62 연습문제
print(int(0.2467 * 12 + 4.159))
print('\n\n')

# p.62 심사문제
print(102*0.6+225)
print('\n\n')

# p.75 연습문제
# a, b, c = map(int, input().split())
a, b, c = -10, 20, 30
print(a+b+c)
print('\n\n')

# p.75 심사문제1
a = 50; b = 100; c = 'None'
print(a); print(b); print(c)
print('\n\n')

# p.75 심사문제2
# a, b, c, d = map(int, input().split())
a , b, c, d = 32, 53, 22, 95
print(int((a+b+c+d)/4))
print('\n\n')

# p.80 연습문제
year, month, day = 2000, 10, 27
hour, minute, second = 11, 43, 59

print(year, month, day, sep='/', end=' ')
print(hour, minute, second, sep=':')
print('\n\n')

# p.81 심사문제
# year, month, day, hour, minute, second = input().split()
year, month, day, hour, minute, second = '2024', '06', '27', '16', '56', '48'

print(year, month, day, sep='-', end='T')
print(hour, minute, second, sep=':')
print('\n\n')

# p.94 연습문제
korean, english, mathematics, science = 92, 47, 86, 81
print((korean>=50) and (english>=50) and (mathematics>=50) and (science>=50))
print('\n\n')

# p.95 심사문제
# korean, english, mathematics, science = map(int, input().split())
korean, english, mathematics, science = 90, 80, 85, 80

print((korean>=90) and (english>80) and (mathematics>85) and (science>=80))
print('\n\n')

# p.100 연습문제
s = """Python is a programming language that lets you work quickly
and
integrate systems more effectively."""

print(s)
print('\n\n')

# p.101 심사문제
s = """'Python' is a 'progamming' langauge
that lets you work quickly
and
integrate systems more effctively."""

print(s)
