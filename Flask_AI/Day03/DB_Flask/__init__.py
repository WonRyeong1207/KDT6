from flask import Flask, render_template

# DataBase 관련
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import config

# create 이전에 설정하는 건가?
DB = SQLAlchemy() # 데이터 베이스와 곤련 매우 높음
MIGRATE = Migrate() # 데이터베이스를 생성해주고 연고ㅕ 리사난 
  
# 사용자 정의 함수
# Application 생성 함수
# - 함수명: create_app <= 이름 변경 불가!!
def create_app():
  
  # 전역변수
  # Flask Web Server instance
  APP = Flask(__name__)
  
  # db 서버 초기화
  APP.config.from_object(config)
  
  # 설정 DB 초기화 및 연결
  DB.init_app(APP)
  MIGRATE.init_app(APP, DB)
  
  
  # URL 즉, 크라이언트 요청 페이지주소를 보여줄 기능 함수
  # def print_page():
  #   return "<h1>Hello, Web</h1>"
  
  # # URL 처리 함수 연결
  # APP.add_url_rule('/', view_func=print_page, endpoint='index')
  
  # 단계를 합친것이 이거
  # @APP.route("/", endpoint="index")
  # def print_page():
  #   return "<h1>Hello, Web</h1>"
  
  # DB 클래스 정의 모듈 loading
  from .models import models
  
  # 라우팅 기능 모듈, URL 처리 모듈
  from .view import main_views, info_views, quest_views, answer_views
  
  APP.register_blueprint(main_views.main_bp)
  APP.register_blueprint(info_views.info_hepta_bp)
  APP.register_blueprint(quest_views.quest_bp)
  APP.register_blueprint(answer_views.answer_bp)
  
  
  
  
  
  

  return APP

if __name__ == '__main__':
    # server 구동
    APP = create_app()
    APP.run()