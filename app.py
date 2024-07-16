from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Sample dataset
data = {
    'age': [65, 70, 80, 75, 85],
    'education_years': [12, 16, 10, 14, 8],
    'mmse_score': [28, 25, 22, 20, 18],
    'moca_score': [26, 24, 21, 19, 16],
    'clock_drawing_score': [5, 4, 3, 2, 1],
    'diagnosis': [0, 1, 1, 1, 1]  # 0: No cognitive decline, 1: Cognitive decline
}

df = pd.DataFrame(data)

X = df[['age', 'education_years', 'mmse_score', 'moca_score', 'clock_drawing_score']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    education_years = int(request.form['education_years'])
    mmse_score = int(request.form['mmse_score'])
    moca_score = int(request.form['moca_score'])
    clock_drawing_score = int(request.form['clock_drawing_score'])

    input_data = np.array([[age, education_years, mmse_score, moca_score, clock_drawing_score]])
    prediction = model.predict(input_data)
    result = 'Cognitive Decline' if prediction[0] == 1 else 'No Cognitive Decline'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
