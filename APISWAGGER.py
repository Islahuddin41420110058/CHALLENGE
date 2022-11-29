#LIBRARY
import pandas as pd 
import pickle, re
import numpy as np
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#BUILD SWAGGER
app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda:'Api Documentation for Deep Learning'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi API untuk Deep Learning'),
    },
    host = LazyString(lambda: request.host)
    )
swagger_config = {
    "headers": [],
    "specs":[
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path":'/flasgger_static',
    "swagger_ui":True,
    "specs_route":"/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

#PARAMETER FEATURE EXTRACTION
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

#LABEL DEFINITION
sentiment = ['negative', 'neutral', 'positive']

#CLEANSING DATA
def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text


def preprocess(text):
    text = lowercase(text) # 1
    text = remove_nonaplhanumeric(text) # 2
    text = remove_unnecessary_char(text) # 2
    return text

#LOAD MODEL CNN
file = open('x_pad_sequences_cnn.pickle','rb')
feature_file_from_cnn = pickle.load(file)
file.close()

model_file_from_cnn = load_model('modelCNN.h5')

#LOAD MODEL RNN
file = open('x_pad_sequences_rnn.pickle','rb')
feature_file_from_rnn = pickle.load(file)
file.close()

model_file_from_rnn = load_model('modelRNN.h5')

#LOAD MODEL LSTM
file = open('x_pad_sequences_lstm.pickle','rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('modellstm.h5')

#ENDPOINT CNN
@swag_from('docs/cnn.yml', methods=['POST'])
@app.route('/cnn', methods=['POST'])
def cnn():

    original_text = request.form.get('text')

    text = [preprocess(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'statuse_code': 200,
        'description': "Result of Sentiment Analysis using CNN",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#ENDPOINT LSTM
@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')

    text = [preprocess(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'statuse_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#ENDPOINT RNN
@swag_from('docs/rnn.yml', methods=['POST'])
@app.route('/rnn', methods=['POST'])
def rnn():

    original_text = request.form.get('text')

    text = [preprocess(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'statuse_code': 200,
        'description': "Result of Sentiment Analysis using RNN",
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data
    
#RUNNING API
if __name__ == '__main__':
  app.run()