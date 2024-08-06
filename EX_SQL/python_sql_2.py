import pymysql

# sql 연결
conn = pymysql.connect(host='localhost', user='test', password='test1111',
                       db='sakila', charset='utf8')

print()
cur = conn.cursor()

query = """
select c.email
from customer as c
    inner join rental as r
    on c.customer_id = r.customer_id
where date(r.rental_date) = (%s)"""

cur.execute(query, '2005-06-16')
rows = cur.fetchall()
for data in rows:
    print(data)
print()

cur.close()
conn.close()