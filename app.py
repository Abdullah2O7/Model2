from flask import Flask
import pickle

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello from the model"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
