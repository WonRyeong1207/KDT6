# Q.41
ticker = 'btc_krw'
ticker = ticker.upper()
print(ticker)
print('\n')

# Q.42
ticker = 'BTC_KRW'
ticker = ticker.lower()
print(ticker)
print('\n')

# Q.43
a = 'hello'
a = a.capitalize()
print(a)
print('\n')

# Q.44
file_name = '보고서.xlsx'
file_name.endswith('xlsx')
print(file_name.endswith('xlsx'))
print('\n')

# Q.45
file_name.endswith(('xlsx', 'xls'))
print(file_name.endswith(('xlsx', 'xls')))
print('\n')

# Q.46
file_name = '2020_보고서_xlsx'
file_name.startswith('2020')
print(file_name.startswith('2020'))
print('\n')

# Q.47
a = 'Hello world'
a, b = a.split()
print(a, b)
print('\n')

# Q.48
ticker = 'btc_krw'
c, d = ticker.split('_')
print(c, d)
print('\n')

# Q.49
data = '2020-05-01'
year, month, day = data.split('-')
print(year, month, day)
print('\n')

# Q.50
data = '039490    '
data = data.rstrip()
print(data * 2)
