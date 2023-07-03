import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go


# python3 -m venv venv

# Linux/macOS: source venv/bin/activate
# Windows: .\venv\Scripts\activate

# pip install -r requirements.txt


def load_data() -> pd.DataFrame:
    """
    Load the Titanic dataset.

    The function checks if the dataset already exists in the local environment,
    if it does, the function reads it directly. If the file doesn't exist, the function
    downloads it from the Stanford University website and saves it as 'titanic.csv' in the
    current working directory.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the Titanic dataset.
    """
    if os.path.isfile('titanic.csv'):
        print('Loading titanic.csv...')
        df = pd.read_csv('titanic.csv')
    else:
        print('Downloading titanic.csv...')
        url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
        download = requests.get(url).content
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        df.to_csv('titanic.csv')
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame for further analysis.

    The function transforms the 'Sex' column from categorical to numerical values.
    It then reduces the DataFrame to only include the columns 'Survived', 'Pclass', 'Sex', 'Age', and 'Fare'.
    It finally drops rows with missing values in these columns.

    Args:
        df (pd.DataFrame): A DataFrame containing the Titanic dataset.

    Returns:
        pd.DataFrame: A preprocessed pandas DataFrame.
    """
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
    df = df.dropna()
    return df


def train_model(df: pd.DataFrame) -> LogisticRegression:
    """
    Trains a Logistic Regression model for survival prediction.

    The function takes a preprocessed DataFrame, separates it into features (X) and target variable (y),
    and performs a train-test split. The model is then trained using the training data.

    Args:
        df (pd.DataFrame): A preprocessed DataFrame containing the Titanic dataset.

    Returns:
        LogisticRegression: A LogisticRegression model trained on the input data.
    """
    print("Training...")
    X = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.values

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict_survival() -> None:
    """
    Provides an interactive interface for user input and predicts the survival probability.

    The function has hardcoded values for 'Passenger Class', 'Gender', and 'Age'
    and uses these inputs to predict the probability of survival using the trained model.
    The survival probability is then printed as a float number.

    Returns:
        None. The function will print the results directly to the console.
    """
    pclass = 2
    sex = 'male'
    age = 42

    # TODO Change the hardcoded values to Streamlit user input (Selectbox/Slider)

    le = LabelEncoder()
    sex = le.fit_transform([sex])[0]

    prediction_proba = model.predict_proba(np.array([pclass, sex, age]).reshape(1, -1))

    survival_proba = prediction_proba[0][1]

    # TODO Create a button to trigger prediction

    print('Survival Probability: ', survival_proba)

    # TODO Display the survival probability in the Streamlit app (with a fancy plotly gauge)


# TODO Create a page that visualizes the age and fare distributions (with interactive altair charts)


# TODO Create a page with an interactive altair chart that shows passengers and their survival


data = load_data()
data = preprocess_data(data)
model = train_model(data)


# TODO Create a page that allows the user to input their own data and get a survival prediction

predict_survival()

# TODO Call the functions created above to display the pages in the Streamlit app
