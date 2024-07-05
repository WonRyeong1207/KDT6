# str 문자열 분리하는 메서드 : split()
# - 기본은 공백기준으로 분리

data = "haha sibal"
result = data.split()
print(result)

phone = '010-1111-2222'
result = phone.split('-')
print(result)

msg = "오늘은 날씨가 좋군요. 내일도 날씨가 좋을까요?"
result = msg.split('.')
print(result)

# 여러개를 한개의 문자열로 합치는 메서드 : join(여러 문자열)
p_result = phone.split('-')
connect = ' '
print(connect.join(p_result))
