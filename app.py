from distutils.log import debug
from flask import Flask , render_template, request
import json
import pandas as pd
import matplotlib.pyplot as plt
import pymongo
from prophet.serialize import model_from_json

app = Flask(__name__)

def getDb():
    CONNECTION_STRING='mongodb+srv://add:v2GQe5gvNiL2IPk0@cluster0.gszdvho.mongodb.net/?retryWrites=true&w=majority'
    try:
        # client = pymongo.MongoClient('localhost',27017)
        client = pymongo.MongoClient(CONNECTION_STRING)
        print("Connection to Mongo success")
    except:
        print("Connection not done")
    mydb = client["climate"]
    return mydb

def insert():
    c = getDb()
    myColl = c["states"]
    with open('./static/data.json',encoding="utf8") as file:
        file_data = json.load(file)
    myColl.insert_many(file_data)
    

@app.route('/')
def index():
    # insert()
    c = getDb()
    mycoll = c["table"].find()
    title = mycoll[0]["title"]
    return render_template('index.html',title = title)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    return render_template('predict2.html')

@app.route('/weather_predict', methods=['GET','POST'])
def weather_predict():
    freq = request.form.get('frequency')[0].lower()
    periods = int(request.form.get('period'))
    what = request.form.get('what')
    if what == 'temperature' : 
        m_max = load_temp_max()
        m_min = load_temp_min()
        df_max_temp = prediction(m_max, periods, freq)
        df_min_temp = prediction(m_min, periods, freq)
        df_max_temp.rename(columns={'yhat':'max'},inplace = True)
        df_min_temp.rename(columns={'yhat':'min'},inplace = True)
        df_min_temp.drop(['ds'], axis= 1,inplace = True)
        df = pd.concat([df_max_temp, df_min_temp],axis= 1, join = 'inner')
        json_data = convert_to_json(df)
        return render_template('predict_temp.html',json_data=json_data)
    
    elif what == 'wind' :
        m = load_wind()
        df = prediction(m,periods, freq)
        json_data = convert_to_json(df)
        return render_template('predict.html',json_data= json_data)
    
    elif what == 'weather' :
        m = load_wind()
        df = prediction(m,periods, freq)
        json_data = convert_to_json(df)
        return render_template('predict.html',json_data=json_data)
    
    m = load_precipitation()
    df= prediction(m,periods, freq)
    json_data = convert_to_json(df)
    return render_template('predict.html', json_data= json_data)

def convert_to_json(df):
    json_data = df.to_json(orient = 'records')
    json_data = json.loads(json_data)
    return json_data


def prediction(m, periods, freq):
    future_dataframe = m.make_future_dataframe(freq=freq, periods=periods)
    predictions = m.predict(future_dataframe)
    df = pd.DataFrame(predictions,columns = ['ds','yhat'])
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')
    df['yhat'] = round(df['yhat'], 3)
    df = df[1460:1460+int(periods)]
    return df

def load_precipitation():
    with open('model.json','r') as fin : 
        m = model_from_json(fin.read())
    return m

def load_temp_max():
    with open('model_temp_max.json','r') as fin :
        m = model_from_json(fin.read())
    return m

def load_temp_min() : 
    with open('model_temp_min.json', 'r') as fin : 
        m = model_from_json(fin.read())
    return m


def load_wind():
    with open('model_wind.json', 'r') as fin : 
        m = model_from_json(fin.read())
    return m

@app.route('/feed')
def feed():
    return render_template('feed.html')

@app.route('/features')
def features():
    l = length()
    mycoll = get_coll("states")
    cName = [0] * l
    for i in range(0,l):
        cName[i] = mycoll[i]["countryName"]
    return render_template('features.html',cname=cName)

def get_coll(s):
    c = getDb()
    return c[s].find()

def length():
    c = getDb()
    mycoll = c["states"].find()
    count = 0
    for i in list(mycoll):
        # print(i)
        count += 1
    return count

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)