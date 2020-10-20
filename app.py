from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    age = int(request.json.get('Age') or 0)
    overall = int(request.json.get('Overall') or 0)
    potential = int(request.json.get('Potential') or 0)
    wage = int(request.json.get('Wage') or 0)
    
    data = [[age, overall, potential, wage]]
    
    prediction = model.predict(data)[0]
    prediction = round(prediction, 2)
    
    formatted_value = '€' + str(round(prediction / 1000, 2)) + 'K' if (prediction / 1000) < 1000 else '€' + str(round(prediction / 1000000, 2)) + 'M'
    
    result = {'value': prediction, 'formatted_value': formatted_value}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)