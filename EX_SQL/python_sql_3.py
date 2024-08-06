import pymysql

# 테이블 생성함수
def create_table(conn, cur):
    try:
        query1 = "drop table if exists customer"
        query2 = """
            create table customer
            (name varchar(10),
             category smallint,
             region varchar(10))
        """
        
        cur.execute(query1)
        cur.execute(query2)
        conn.commit()
        print('Table 생성 완료')
         
    except Exception as e:
        print(e)

def main():
    # sql 연결
    conn = pymysql.connect(host='localhost', user='test', password='test1111',
                        db='sqlclass_db', charset='utf8')

    print()
    cur = conn.cursor()

    create_table(conn, cur)
    
    cur.close()
    conn.close()
    print('Database 연결종료')
    

main()