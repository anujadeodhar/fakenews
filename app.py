# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
#filename = 'fake-news-model.pkl'
classifier = pickle.load(open(r'C:/Users/Anuja/PycharmProjects/Fake-News-Classfier-master - Copy/fake-news-model.pkl', 'rb'))
cv = pickle.load(open(r'C:/Users/Anuja/PycharmProjects/Fake-News-Classfier-master - Copy/cv-transform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

#predicting
#1-false 0-true
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = cv.transform(data).toarray()
        my_prediction = classifier.predict(vector)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
