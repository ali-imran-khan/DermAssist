from flask import Flask
from uploader import upload_blueprint

app = Flask(__name__)
app.register_blueprint(upload_blueprint)