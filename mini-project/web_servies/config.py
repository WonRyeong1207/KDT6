# ORM 적용을 위한 파일
import os

BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'web_sev.db'

DB_MYSQL_URL = 'mysql+pymysql://root:1234@localhost:3306/web_servies'
# DB_MYSQL_URL = 'mysql+pymysql://root:1234@172.20.97.151:3306/web_servies'
# DB_MYSQL_URL = 'mysql+pymysql://root:1234@172.20.97.151:3306/web_servies?auth_plugin=caching_sha2_password'
DB_MARIA_URL = 'mariadb+mariadbconnector://root:root!@127.0.0.1:3308/db_ai'  # 수정: :: -> :

SQLALCHEMY_DATABASE_URI = DB_MYSQL_URL
# 'sqlite:///{}'.format(os.path.join(BASE_DIR, DB_NAME))

SQLALCHEMY_TRACK_MODIFICATIONS = False
