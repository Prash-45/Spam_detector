from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model (1).pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # Transform text
    transformed_message = tfidf.transform([message])

    # Prediction
    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "Spam Message 🚨"
    else:
        result = "Not Spam ✅"

    return render_template(
        'index.html',
        prediction_text=result,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)