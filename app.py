from distutils.log import debug
import pickle
from flask import Flask , render_template, request

app = Flask(__name__)
# model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)