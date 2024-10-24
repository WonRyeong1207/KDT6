from Flask import DB

class korean_food(DB.Model):
    __tablename__ = 'korean_food'  # 테이블 이름을 명시적으로 설정
    id = DB.Column(DB.Integer, primary_key=True)  # 기본 키로 설정
    subtitle = DB.Column(DB.String(50))
    title = DB.Column(DB.String(50))
    img_url = DB.Column(DB.String(128))
    level = DB.Column(DB.String(50))
    level_code = DB.Column(DB.Integer)
    link = DB.Column(DB.String(128))
    time = DB.Column(DB.String(50))
    element = DB.Column(DB.String(8192))
    feature = DB.Column(DB.String(4096))
    
# class japanfood(DB.Model):
#     name = DB.Column(DB.String(50), nullable=False)
#     title = DB.Column(DB.String(50), nullable=False)
#     food =  DB.Column(DB.String(50), nullable=False)
#     link = DB.Column(DB.String(128), nullable=False)
    
# class usafood(DB.Model):
#     name = DB.Column(DB.String(50), nullable=False)
#     title = DB.Column(DB.String(50), nullable=False)
#     food =  DB.Column(DB.String(50), nullable=False)
#     link = DB.Column(DB.String(128), nullable=False)
    
    