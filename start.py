import spacy
import wikipedia
import pytextrank
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle

with open("db.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('mtx')

    # load tokenizer object
    with open('mtx_t.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('mtx_l.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20

    while True:
        print("interact>> " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                if np.random.choice(i['responses']) == "ws":
                    nlp = spacy.load("en_core_web_sm")
                    nlp.add_pipe("textrank")
                    doc = nlp(inp)
                    for phrase in doc._.phrases[:1]:
                        try:
                            p = wikipedia.summary(phrase.text)
                            print(">> " + p)
                        except wikipedia.DisambiguationError as e:
                            break
                        except wikipedia.PageError as pe:
                            break
                    break
                print(">> " + np.random.choice(i['responses']))

chat()
