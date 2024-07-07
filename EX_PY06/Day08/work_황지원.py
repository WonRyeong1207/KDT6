# 24, 29, 30장

# p.303 연습문제
path = 'C:\\Users\\dojang\\AppData\\Local\\Programs\\Python\\Python36-32\\python.exe'

filename = path[path.rfind('\\')+1:]

print(filename)
print()

# 아니면..
x = path.split('\\')
filename = x[-1] # 파일 경로에서 파일을 찾는 경우 항상 마지막이 파일명
print(filename)
print()

# p.305 심사문제 1
data = input()
# data1 = "the grown-ups' response, this time, was to advise me to lay aside mt drawings of boa constrictors, whether from the inside or outside, and devote myself instead to geography, history, arithmetic, and grammar."
# data2 = "That is why, at the, age of six, I gave up what might have been a magnificent career as a painter."
# data3 = "I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two."
# data4 = "Grown-ups naver understand anything by themselves, and it is tiresome for children to be always and forever explaining thing to the."
# data = data1 + data2 + data3 + data4
# 한 문장이지만 너무 길어...

other = ['them', 'there', 'their'] # 3인칭 복수형을 제외하려함. whe'the'r와 같은 경우는 포함
con_num = data.count('the') # 수를 셈. 이때는 제외하려는것도 포함됨.

for i in range(len(other)): # 제외하러 감.
    current = other[i] # current 현재 진행 상태를 표현하려고 씀.
    
    if (data.count(current)):
        con_num -= data.count(current)
        
    # 한줄로 표현하면
    # if (data.count(other[i])): con_num -= data.count(current)
        
print(con_num)
print()

# p.305~306 심사문제2
price_list = list(map(str, input().split(';')))
# price_input = '51900;83000;158000;367500;250000;59200;128500;1304000'
# price_list = list(price_input.split(';'))
# print(price_list)

current_list = []
for i in price_list:
    current = list(i)
    
    if 3 < len(current) <= 6:
        current.reverse()
        current.insert(3, ',')
        current.reverse()
        current_list.append(''.join(current))
    
    elif len(current) > 6:
        current.reverse()
        current.insert(3, ',')
        current.insert(7, ',')
        current.reverse()
        current_list.append(''.join(current))
    else:
        current_list.append(i)
# print(current_list)

carry_list = []
for i in range(len(current_list)):
    current = current_list[i]
    carry_list.append('{0:>9}'.format(current))
price_list = sorted(carry_list, reverse=True)
# print(price_list)

for i in range(len(price_list)):
    # pass
    print(''.join(price_list[i]))
print()

# p.384 연습문제
x = 10
y = 3

def get_quotient_remainder(x, y):
    return (x//y), (x%y)

quotient, remainder = get_quotient_remainder(x,y)
print('몫: {0}, 나머지: {1}'.format(quotient, remainder))
print()

# p.384~385 심사문제
x, y = map(int, input().split())
# x, y = 10, 20

def calc(n1, n2):
    if n2 != 0:
        return (n1+n2), (n1-n2), (n1*n2), (n1/n2)
    else:
        return (n1+n2), (n1-n2), (n1*n2), None
    
a, s, m, d = calc(x, y)
print("덧셈: {0}, 뺄셈: {1}, 곱셈: {2}, 나눗셈: {3}".format(a, s, m, d))
print()

# p.397 연습문제
korean, english, mathematics, science = 100, 86, 81, 91

def get_max_score(*args):
    return max(args)

max_score = get_max_score(korean, english, mathematics, science)
print('높은 점수:', max_score)
max_score = get_max_score(english, science)
print('높은 점수:', max_score)
print()

# p.398 심사문제
korean, english, mathematics, science = map(int, input().split())
# korean, english, mathematics, science = 76, 82, 89, 84

def get_min_max_score(*args):
    return min(args), max(args)

def get_average(korean=0, english=0, mathematics=0, science=0):
    cnt = 0
    if korean:
        cnt += 1
    if english:
        cnt += 1
    if mathematics:
        cnt += 1
    if science:
        cnt += 1
    return (korean + english + mathematics + science)/cnt

# type error
# def get_average(*args):
#     return sum(args)/len(args)

min_score, max_score = get_min_max_score(korean, english, mathematics, science)
average_score = get_average(korean=korean, english=english, mathematics=mathematics, science=science)
print('낮은 점수: {0: .2f}, 높은 점수: {1: .2f}, 평균 점수: {2: .2f}'.format(min_score, max_score, average_score))

in_score, max_score = get_min_max_score(english, mathematics)
average_score = get_average(english=english, science=science)
print('낮은 점수: {0: .2f}, 높은 점수: {1: .2f}, 평균 점수: {2: .2f}'.format(min_score, max_score, average_score))
