# Flask Framework에서 '/' URL에 대한 라우팅 처리
# - 파일명: main_views.py

from flask import Blueprint, render_template
from DB_Flask.models.models import Question

# Blueprint instance
# http://127.0.0.1:5000/
quest_bp = Blueprint('quest', import_name='__name__', url_prefix='/', template_folder='templates')

# Routing Functions
# URL 처리
@quest_bp.route('/question/', endpoint='list')   
# endpoint: rul끝단. 플라스크에서의 의미 url의 끝단이 아닌 그걸 처리하는 함수의 별칭.
# 함수명을 외부에 노출 시키지 않을 수 있음. 내부적으로 함수명을 바꿀 수 있음.
def quest_list():
    q_list = Question.query.all()
    return render_template('question_list.html', question_list=q_list)


@quest_bp.route('/question/detail/<int:q_id>', endpoint='detail')
def quest_detail(q_id):
    q = Question.query.get(q_id)
    return render_template('question_detail.html', question=q)