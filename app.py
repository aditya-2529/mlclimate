from distutils.log import debug
import pickle
from flask import Flask , render_template, request

app = Flask(__name__)
model = pickle.load(open('ml\climate\model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/weather_predict', methods=['GET','POST'])
def weather_predict():
    prediction = model.predict([[0.1,15,28,10,0,0,0,1]])
    output = round(prediction[0], 20)
    return render_template('predict.html',prediction_text=f'Max temperature is: {output}')

@app.route('/feed')
def feed():
    return render_template('feed.html')

# @app.route('/predict', methods=['GET','POST'])
# def predict():
    # output = round(prediction[0], 2)
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
