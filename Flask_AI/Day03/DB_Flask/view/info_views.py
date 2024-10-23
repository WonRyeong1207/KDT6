# Flask Framework에서 '/' URL에 대한 라우팅 처리
# - 파일명: main_views.py

from flask import Blueprint, render_template

# Blueprint instance
# http://127.0.0.1:5000/
info_hepta_bp = Blueprint('info_hepta', import_name='__name__', url_prefix='/', template_folder='templates')

# Routing Functions
# URL 처리
@info_hepta_bp.route('/info_hepta/', endpoint='hepta')   
# endpoint: rul끝단. 플라스크에서의 의미 url의 끝단이 아닌 그걸 처리하는 함수의 별칭.
# 함수명을 외부에 노출 시키지 않을 수 있음. 내부적으로 함수명을 바꿀 수 있음.
def info_hepta():
    return render_template('info_hepta.html')
