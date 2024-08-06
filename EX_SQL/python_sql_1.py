import pymysql
import pandas as pd
import pymysql.cursors

# sql 연결
conn = pymysql.connect(host='localhost', user='test', password='test1111',
                       db='sakila', charset='utf8') # 또는 여기에 , cursorclass=pymysql.cursor.DictCursor

print()
cur = conn.cursor(pymysql.cursors.DictCursor)   # DataFrame의 column들을 같이 리턴
cur.execute('select * from language')

desc = cur.description  # 헤더 정보를 가져옴
for i in range(len(desc)):
    print(desc[i][0], end=' ')
print()

row = cur.fetchall()    # 모든 데이터를 가져옴
for data in row:
    print(data)
print()

language_df = pd.DataFrame(row)     # 판다스 데이터프레임
print(language_df)

cur.close()
conn.close()    # 데이터 베이스 연결종료
