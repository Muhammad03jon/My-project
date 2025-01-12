import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Заголовок приложения
st.title('Прогнозирование диабета с помощью CatBoost')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/My-project/refs/heads/master/diabetes%20(2).csv"
df = pd.read_csv(file_path)

# Раздел для отображения данных
with st.expander('Данные'):
    st.write("Полный набор данных:")
    st.dataframe(df)

# Раздел для ввода данных пользователем
with st.sidebar:
    st.header("Введите признаки:")
    age = st.slider('Возраст', 21, 100, 30)
    gender = st.selectbox('Пол', ('Мужской', 'Женский'))
    bmi = st.slider('Индекс массы тела (BMI)', 10.0, 70.0, 25.0)
    sbp = st.slider('Систолическое артериальное давление (SBP)', 80, 200, 120)
    dbp = st.slider('Диастолическое артериальное давление (DBP)', 40, 120, 80)
    fpg = st.slider('Глюкоза натощак (FPG)', 50, 200, 100)
    chol = st.slider('Общий холестерин (Chol)', 100, 400, 200)
    tri = st.slider('Триглицериды (Tri)', 50, 400, 150)
    hdl = st.slider('Холестерин высокой плотности (HDL)', 20, 100, 50)
    ldl = st.slider('Холестерин низкой плотности (LDL)', 50, 200, 100)
    alt = st.slider('Аланинаминотрансфераза (ALT)', 10, 100, 20)
    bun = st.slider('Мочевина (BUN)', 5, 50, 20)
    ccr = st.slider('Креатининовый клиренс (CCR)', 30, 150, 60)
    ffpg = st.slider('Глюкоза в крови на пальце (FFPG)', 50, 200, 100)
    smoking = st.selectbox('Курите ли вы?', ('Нет', 'Да'))
    drinking = st.selectbox('Употребляете ли вы алкоголь?', ('Нет', 'Да'))
    family_history = st.selectbox('Есть ли в семье диабет?', ('Нет', 'Да'))

# Преобразование пользовательского ввода в DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Мужской' else 0],
    'BMI': [bmi],
    'SBP': [sbp],
    'DBP': [dbp],
    'FPG': [fpg],
    'Chol': [chol],
    'Tri': [tri],
    'HDL': [hdl],
    'LDL': [ldl],
    'ALT': [alt],
    'BUN': [bun],
    'CCR': [ccr],
    'FFPG': [ffpg],
    'Smoking': [1 if smoking == 'Да' else 0],
    'Drinking': [1 if drinking == 'Да' else 0],
    'FamilyHistory': [1 if family_history == 'Да' else 0]  # Исправлено название столбца на 'FamilyHistory'
})

# Подготовка данных для обучения
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание модели CatBoost
model = CatBoostClassifier(
    iterations=150,
    l2_leaf_reg=6,
    learning_rate=0.05,
    max_depth=6,
    rsm=0.3,
    verbose=0
)

# Обучение модели
model.fit(X_train_scaled, y_train)

# Оценка модели
train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

st.subheader("Точность модели")
st.write(f"Точность на обучающей выборке: {train_accuracy:.2f}")
st.write(f"Точность на тестовой выборке: {test_accuracy:.2f}")

# --- Матрица ошибок ---
cm = confusion_matrix(y_test, model.predict(X_test_scaled))
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm, cmap=plt.cm.Blues)

# Отображение в Streamlit
st.subheader("Матрица ошибок")
st.pyplot(fig_cm)

# Прогнозирование для пользовательских данных
if st.button('Предсказать'):
    # Стандартизация пользовательских данных
    input_data_standardized = scaler.transform(input_data)

    # Прогнозирование с использованием стандартизированных данных
    prediction = model.predict(input_data_standardized)
    prediction_proba = model.predict_proba(input_data_standardized)

    # Отображение результатов
    st.subheader("Результаты предсказания")
    st.write(f"Вероятность отсутствия диабета: {prediction_proba[0][0]:.2f}")
    st.write(f"Вероятность наличия диабета: {prediction_proba[0][1]:.2f}")
    st.success(f"Предсказание: {'Диабет' if prediction[0] == 1 else 'Нет диабета'}")


# Визуализация данных
st.subheader('Визуализация данных')
fig = px.scatter(
    df,
    x='BMI',
    y='Age',
    color='Diabetes',
    title='Индекс массы тела (BMI) и возраст по наличию диабета'
)
st.plotly_chart(fig)

