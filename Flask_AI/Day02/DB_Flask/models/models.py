# database table define class

from DB_Flask import DB

# Question class
# PK: id
class Question(DB.Model):
    # define columns
    id = DB.Column(DB.Integer, primary_key=True)
    subject = DB.Column(DB.String(200), nullable=False)
    content = DB.Column(DB.Text(), nullable=False)
    create_data = DB.Column(DB.DateTime(), nullable=False)
    
# Answer class
# PK: id
# FK: question_id
class Answer(DB.Model):
    # define columns
    id = DB.Column(DB.Integer, primary_key=True)
    question_id = DB.Column(DB.Integer, DB.ForeignKey("question.id", ondelete='CASCADE'))
    question = DB.relationship('Question', backref=DB.backref('answer_set',))
    content = DB.Column(DB.Text(), nullable=False)
    create_data = DB.Column(DB.DateTime(), nullable=False)
    
