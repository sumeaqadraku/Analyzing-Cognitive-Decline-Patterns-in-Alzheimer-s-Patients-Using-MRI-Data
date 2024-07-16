import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_and_prepare_data():
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
    return df

def train_model(df):
    X = df[['age', 'education_years', 'mmse_score', 'moca_score', 'clock_drawing_score']]
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'analysis/model.pkl')  # Save the model
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def create_visualizations(df):
    sns.histplot(df['mmse_score'])
    plt.title('Distribution of MMSE Scores')
    plt.savefig('static/mmse_histogram.png')
    plt.clf()

    sns.boxplot(x='diagnosis', y='mmse_score', data=df)
    plt.title('MMSE Scores by Diagnosis')
    plt.savefig('static/mmse_boxplot.png')
    plt.clf()

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('static/correlation_matrix.png')
    plt.clf()

if __name__ == "__main__":
    df = load_and_prepare_data()
    model, X_test, y_test = train_model(df)
    accuracy, report = evaluate_model(model, X_test, y_test)
    create_visualizations(df)
    print(f'Model Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')
