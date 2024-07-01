# set
# - 연산자

d1 = {1, 3, 5, 7}
d2 = {2, 4, 6, 8}

# plus
# print(d1 + d2)
# 시퀀스 타입이 아니기 때문에 지원하지 않음.
print(d1.union(d2)) # .union 새로운 집합, 합집합
d3 = {1, 2, 3, 4, 6, 7, 4, 2}
print(d1.union(d3))
print(d1.union(d2), d1|d2)

# 교집합
print(d1.intersection(d3))
print(d2.intersection(d3), d2&d3)

# 차집합
print(d1.difference(d2))
print(d2.difference(d3), d2-d3)
print(d3-d2)


# 이외에도 다양한 것이 존재함.