import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

st.write("""
# MSCI 436 - Group 6's Decision Support System 

This app aims to improve students' study habits using Linear Regression!
\n
""")

st.sidebar.header('Adjust Student Input Parameters')
st.sidebar.write('Copy URL for Legend: shorturl.at/cfUY7')

def user_input_features():
    age = st.sidebar.slider('Student\'s Age', 15, 22, 15)
    medu = st.sidebar.slider('Mother\'s Education', 0, 4, 0)
    fedu = st.sidebar.slider('Father\'s Education', 0, 4, 0)
    traveltime = st.sidebar.slider('Home to School Travel Time', 1, 4, 1)
    studytime = st.sidebar.slider('Weekly Study Time', 1, 4, 1)
    failures = st.sidebar.slider('Number of Past Class Failures', 0, 3, 0)
    famrel = st.sidebar.slider('Family Relationship Quality', 1, 5, 1)
    freetime = st.sidebar.slider('Free Time After School', 1, 5, 1)
    goout = st.sidebar.slider('Going Out with Friends', 1, 5, 1)
    dalc = st.sidebar.slider('Workday Alcohol Consumption', 1, 5, 1)
    walc = st.sidebar.slider('Weekend Alcohol Consumption', 1, 5, 1)
    health = st.sidebar.slider('Current Health Status', 1, 5, 1)
    absences = st.sidebar.slider('Number of School Absences', 0, 93, 0)



    data = {
        'age': age,
        'Medu': medu,
        'Fedu': fedu,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': dalc,
        'Walc': walc,
        'health': health,
        'absences': absences}

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('Additional Preprocessing')
features_nonused = st.multiselect('Select variables NOT to be used', df.columns)


st.subheader('User Input Parameters')
show_table = st.checkbox('Show the dataset as table data', value=True)
if show_table:
    st.dataframe(df)

dataset = pd.read_csv('./student-por.csv')

grades_pass_fail = []
for index, row in dataset.iterrows():
    if row['G3'] >= 10:
        grades_pass_fail.append(1)
    else:
        grades_pass_fail.append(0)

grades_pass_fail_series = pd.Series(grades_pass_fail)
dataset["pass_fail"] = grades_pass_fail_series

grades_erasmus_label_encoded = []
for index, row in dataset.iterrows():
    if row['G3'] >= 16:
        grades_erasmus_label_encoded.append(1)
    elif row['G3'] == 15 or row['G3'] == 14:
        grades_erasmus_label_encoded.append(2)
    elif row['G3'] == 12 or row['G3'] == 13:
        grades_erasmus_label_encoded.append(3)
    elif row['G3'] == 10 or row['G3'] == 11:
        grades_erasmus_label_encoded.append(4)
    elif row['G3'] <= 9:
        grades_erasmus_label_encoded.append(5)

grades_erasmus_label_encoded_series = pd.Series(grades_erasmus_label_encoded)
dataset["Erasmus_Grade_Label_Encoded"] = grades_erasmus_label_encoded_series

grades_erasmus = []
for index, row in dataset.iterrows():
    if row['G3'] >= 16:
        grades_erasmus.append('A')
    elif row['G3'] == 15 or row['G3'] == 14:
        grades_erasmus.append('B')
    elif row['G3'] == 12 or row['G3'] == 13:
        grades_erasmus.append('C')
    elif row['G3'] == 10 or row['G3'] == 11:
        grades_erasmus.append('D')
    elif row['G3'] <= 9:
        grades_erasmus.append('F')

grades_erasmus_series = pd.Series(grades_erasmus)
dataset["Erasmus_Grade"] = grades_erasmus_series

X = dataset[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout',
                   'Dalc', 'Walc', 'health', 'absences']].values
X_userparams = df.values
y = dataset.iloc[:, -4].values

dataset = dataset.drop(columns=features_nonused)
df = df.drop(columns=features_nonused)

st.subheader('Configure Training Parameters')
left_column, right_column = st.columns(2)

test_size = left_column.number_input(
    'Validation Dataset Size:',
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

random_seed = right_column.number_input(
    'Set Random Seed Value:',
    min_value=0,
    value=0,
    step=1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

st.subheader('View Feature Importance')
importance = regressor.coef_
for i, v in enumerate(importance):
    v = "{:.2f}".format(v)
fig = plt.figure(figsize=(4, 4))
plt.xlabel('Feature #')
plt.ylabel('Score')
plt.bar([i for i in range(len(importance))], importance)
st.pyplot(fig)

index_max = np.argmax(importance)
st.write('The factor most significantly impacting your grade is ', df.columns[index_max], '.')

expected_grade = regressor.predict(X_userparams)

st.subheader('Final Expected Grade:')
st.write('Your expected grade is: ', round((expected_grade.astype(int)[0] / 20.0) * 100, 3), '%')
