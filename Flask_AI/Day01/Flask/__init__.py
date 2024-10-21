# 플라스크는 처음이라...
# cgi랑은 또다른 방식이라던데..

# Flask FrameWork Web 구동 방법
# - file_name: app.py


# load module
# import importlib.metadata
import flask
import pandas as pd

from flask import Flask, render_template
# import importlib

# print(f"pandas version: {pd.__version__}")
# print(f"flask version: ")
# print(importlib.metadata.version("flask"))

# 사용자 정의 함수
def create_app():
  
  # 전역변수
  # Flask Web Server instance
  APP = Flask(__name__)

  # 라우팅 기능 함수
  # @Flask Web Server 인스턴스변수명.route("url")
  # http://127.0.0.1:5000

  # set FLASK_APP=my_app.py
  # 임시적으로 변경하는 방법


  test_index =  """
  <body style='background-color: skyblue;'>
    <h3>Hello, Flask!<h3>
      <lable>my list</label>
      <ul>
        <li>not thing</li>
      </ul>
  </body>
  """

  # http://127.0.0.1:5000
  @APP.route("/") # 리눅스: root - 어떤 sw의 시작점/저장소
  def index():
      # return "<h1>Hello, Flask</h2>"
      # return test_index
      return render_template("index.html")

  info_index = """
  <body style='background-color: yellow;'>
    <h3>heptagram Infomation</h3>
    <label>heapagram</label>
    <img src="{{url_for('static', filename='image/heptagram_face.png')}}" alt="hetagram_face"><br>
    <ul>
      <li><strong>Name</strong>: Hetagram</li>
      <li><strong>Birthday</strong>: April 23th</li>
    </ul>
    
  </body>
  """

  # http://127.0.0.1:5000/info
  # http://127.0.0.1:5000/info/

  @APP.route('/info')
  @APP.route('/info/')
  def info():
      # return info_index
      return render_template("info_hepta.html")
      


  # http://127.0.0.1:5000/info/name=nana
  @APP.route("/info/<name>")
  def print_info(name):
      # f = f"""
      # <h3>{name}'s Infomation</h3>
      #   hello~ {name}!
      # """
      # return f
      return render_template("info.html", name=name)

  # http://127.0.0.1:5000/info/정수형
  @APP.route("/info/<int:age>")
  def check_age(age):
      f = f"""
      <h3>{age}'s Infomation</h3>
        age: {age}!
      """
      return f

  # http://127.0.0.1:5000/go
  @APP.route("/go")
  @APP.route("/info/go")
  def go_home():
      return APP.redirect('/')

  return APP

if __name__ == '__main__':
    # server 구동
    APP = create_app()
    APP.run()