# Flask Framework에서 '/' URL에 대한 라우팅 처리
# - 파일명: main_views.py

from datetime import datetime
from flask import Blueprint, render_template, url_for, request
from DB_Flask.models.models import Answer, Question
from werkzeug.utils import redirect
from DB_Flask import DB

# Blueprint instance
# http://127.0.0.1:5000/
answer_bp = Blueprint('answer', import_name='__name__', url_prefix='/answer', template_folder='templates')

# Routing Functions
# URL 처리
@answer_bp.route('/answer/create/<int:q_id>', methods=['POST'], endpoint='create')   
# endpoint: rul끝단. 플라스크에서의 의미 url의 끝단이 아닌 그걸 처리하는 함수의 별칭.
# 함수명을 외부에 노출 시키지 않을 수 있음. 내부적으로 함수명을 바꿀 수 있음.
def answer_create(q_id):
    
    question = Question.query.get_or_404(q_id)
    content = request.form['content']
    answer = Answer(content=content, create_date=datetime.now())
    question.answer_set.append(answer)
    DB.session.commit()
    
    return redirect(url_for('quest.detail', q_id=question.id))