from distutils.log import debug
import pickle
from flask import Flask , render_template, request

app = Flask(__name__)
model = pickle.load(open('ml\climate\model.pkl','rb'))

@app.route('/')
def index():
    prediction = model.predict([[0.1,15,28,10,0,0,0,1]])
    output = round(prediction[0], 20)
    return render_template('index.html',prediction_text=f'{output}')

# @app.route('/predict', methods=['GET','POST'])
# def predict():
    # output = round(prediction[0], 2)
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
