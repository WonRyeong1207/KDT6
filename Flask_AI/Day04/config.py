# ORM 적용을 위한 파일
import os

BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'DB_Flask.db'

SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, DB_NAME))

DB_MYSQL_URL = 'mysql+pymysql://root:1234@local:3306/testdb'
DB_MARIA_URL = 'mariadb+mariadbconnector://root:root!@127.0.0.1::3308/db_ai'
# //뒤에는 예시

SQLALCHEMY_TRACK_MODIFICATIONS = False

