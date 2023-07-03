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

# Run with:
# streamlit run main.py


@st.cache_data
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


@st.cache_data
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


@st.cache_data
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

    The function allows a user to input 'Passenger Class', 'Gender', and 'Age' through interactive
    widgets and uses these inputs to predict the probability of survival using the trained model.
    The survival probability is then displayed as a number and a gauge chart.

    Returns:
        None. The function will output the results directly to the Streamlit interface.
    """
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Gender', ['male', 'female'])
    age = st.slider('Age', 0, 100, 25)

    le = LabelEncoder()
    sex = le.fit_transform([sex])[0]

    prediction_proba = model.predict_proba(np.array([pclass, sex, age]).reshape(1, -1))

    if st.button('Predict'):
        survival_proba = prediction_proba[0][1]
        st.write('Survival Probability: ', survival_proba)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=survival_proba,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "darkblue"}},
            title={'text': "Survival Probability"}))
        st.plotly_chart(fig)


def visualize_data(df: pd.DataFrame) -> None:
    st.header('Data Visualizations')

    if st.checkbox('Show Age Distribution'):
        age_chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
            alt.X("Age:Q", bin=alt.Bin(maxbins=20)),
            y='count()',
        ).properties(
            title='Age Distribution'
        )
        st.altair_chart(age_chart, use_container_width=True)

    if st.checkbox('Show Fare Distribution'):
        fare_chart = alt.Chart(df).mark_bar(opacity=0.7, color='red').encode(
            alt.X("Fare:Q", bin=alt.Bin(maxbins=20)),
            y='count()',
        ).properties(
            title='Fare Distribution'
        )
        st.altair_chart(fare_chart, use_container_width=True)


def interactive_plot(df: pd.DataFrame) -> None:
    st.header('Interactive Plot')
    pclass_to_filter = st.slider('Passenger Class', 1, 3, 1)
    filtered_df = df[df['Pclass'] == pclass_to_filter]

    chart = alt.Chart(filtered_df).mark_circle().encode(
        x='Age:Q',
        y='Fare:Q',
        color=alt.condition(
            alt.datum.Survived == 1,
            alt.value('green'),
            alt.value('red')
        ),
        tooltip=['Age:Q', 'Fare:Q', 'Survived:N', 'Pclass:N']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


data = load_data()
data = preprocess_data(data)
model = train_model(data)

st.title('Titanic Survival Prediction')

page = st.sidebar.selectbox('Choose a page', ['Home', 'Visualizations', 'Interactive Plot'])

if page == 'Home':
    st.write('A simple Streamlit app for exploring the Titanic dataset and predicting survival.')

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    st.subheader('Survival by Passenger Class and Gender')
    pivot_df = data.pivot_table(index=['Pclass', 'Sex'], values='Survived', aggfunc=np.mean)
    st.write(pivot_df)

    predict_survival()

elif page == 'Visualizations':
    visualize_data(data)

elif page == 'Interactive Plot':
    interactive_plot(data)
