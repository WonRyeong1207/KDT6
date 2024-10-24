# 한번 시도를 해보자...

from flask import Flask, render_template

# DataBase 관련
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import config

# create 이전에 설정하는 건가?
DB = SQLAlchemy() # 데이터 베이스와 곤련 매우 높음
MIGRATE = Migrate() # 데이터베이스를 생성해줌
    
def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    
    DB.init_app(app)
    MIGRATE.init_app(app, DB)
    
    from .models import models
    
    from .view import main_views, ko_food_views
    
    app.register_blueprint(main_views.main_bp)
    app.register_blueprint(ko_food_views.ko_food_bp)
    
    
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()