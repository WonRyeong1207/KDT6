# 12장부터 15장까지

# p.149 연습문제
camille = {
    'health':575.6,
    'health_regen':1.7,
    'mana':338.8,
    'mana_regen':1.63,
    'melee':125,
    'attack_damage':60,
    'attack_speed':0.625,
    'armor':26,
    'magic_resistance':32.1,
    'movement_spped':340
}

print(camille['health'])
print(camille['movement_spped'])
print('\n')

# p.150 심사문제
key_list = list(input().split())
value_list = list(map(float,input().split()))
# key_list = ['health', 'health_regen', 'mana', 'mana_regen']
# value_list = [575.6, 1.7, 338.8, 1.63]

character = dict(zip(key_list, value_list))
print(character)
print('\n')

# p.164 연습문제
x = 5

if (x != 10):
    print('ok!')
print('\n')
    
# p.165 심사문제
price = int(input())
coopon = input()
# price = 27000
# coopon = 'Cash3000'

if (coopon == 'Cash3000'):
    price = price - 3000
if (coopon == 'Cash5000'):
    price = price - 5000
print(price)
print('\n')

# p.174 연습문제
written_test = 75
coding_test = True

if ((written_test >= 80) and (coding_test == True)):
    print('합격')
else:
    print('불합격')
print('\n')

# p.174 ~ 175 심사문제
k, e, m, s = map(int, input().split())
# k, e, m, s = 89, 72, 93, 82

if ((k<0) or (k>100) or (e<0) or (e>100) or (m<0) or (m>100) or (s<0) or (s>100)):
    print('잘못된 점수')
if (((k+e+m+s)/4) >= 80):
    print("합격")
else:
    print("불합격")
    
# p.180 연습문제
x = int(input())
# x = 5

if ((11 <= x) and (x <= 20)):
    print('11 ~ 20')
elif ((12 <= x) and (x <= 30)):
    print('21 ~ 30')
else:
    print("아무것도 해당되지 않음")
print('\n')

# p.181 심사문제
age = int(input())
# age = 24
balance = 9000

if ((7 <= age) and (age < 13)):
    balance = balance - 650
elif ((13 <= age) and (age < 19)):
    balance = balance - 1050
else:
    balance = balance - 1250
print(balance)