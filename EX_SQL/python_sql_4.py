import pymysql

# sql 연결
conn = pymysql.connect(host='localhost', user='test', password='test1111',
                    db='sqlclass_db', charset='utf8')

print()
curs = conn.cursor()

# name 맞는데?
sql = """insert into customer(name, category, region)
        values (%s, %s, %s)"""

curs.execute(sql, ('홍길동', 1, '서울'))
curs.execute(sql, ('이연수', 2, '서울'))
conn.commit()

print('INSERT 완료')

curs.execute('select * from customer')
rows = curs.fetchall()
print(rows)

curs.close()
conn.close()