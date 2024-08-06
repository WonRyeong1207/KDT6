import pymysql

# sql 연결
conn = pymysql.connect(host='localhost', user='test', password='test1111',
                    db='sqlclass_db', charset='utf8')

print()
curs = conn.cursor()

sql = """insert into customer(name, category, region)
        values (%s, %s, %s)"""
data = (('홍진우', 1, '서울'),
        ('강지수', 2, '부산'),
        ('김청진', 1, '대구'),)

curs.executemany(sql, data)
conn.commit()
print('executemany() 완료')

curs.close()
conn.close()