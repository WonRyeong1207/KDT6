# 한번 연결 해봅시다!

from flask import Flask, render_template

def create_app():
    
    app = Flask(__name__)
    
    
    @app.route('/')
    @app.route('/my_project')
    def main():
        return render_template("my_project.html")
    
    @app.route('/ml')
    def ml():
        return render_template("ml.html")
    
    @app.route('/dl')
    def dl():
        return render_template("dl.html")
    
    @app.route('/vision')
    def vision():
        return render_template("vision.html")
    
    @app.route('/nlp')
    def nlp():
        return render_template("nlp.html")
    
    return app

if __name__=='__main__':
    app = create_app()
    app.run()