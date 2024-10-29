from flask import Flask, render_template

# 이번에는 DB 안 쓸거임
def create_app():
    app = Flask(__name__)
    
    from .views import main_views, model_views
    app.register_blueprint(main_views.main_bp)
    app.register_blueprint(model_views.model_bp)
    
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()