import streamlit as st
import pandas as pd
import json
import os
import hashlib
import requests
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# === Файлы ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"
RETRAIN_METRICS_FILE = "retrain_metrics.json"
TRAINING_LOG_FILE = "training_log.txt"

# === Работа с пользователями ===
def load_users():
    return json.load(open(USER_DB, "r", encoding="utf-8")) if os.path.exists(USER_DB) else {}

def save_users(users):
    json.dump(users, open(USER_DB, "w", encoding="utf-8"), indent=4)

def register_user(username, password):
    users = load_users()
    if username in users:
        return "Пользователь уже существует"
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    save_users(users)
    return "Регистрация успешна"

def login_user(username, password):
    users = load_users()
    return username in users and users[username] == hashlib.sha256(password.encode()).hexdigest()

# === Обучение модели ===
def train_ml_model():
    st.subheader("Обучение модели")
    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        st.error("Файл датасета не найден.")
        return

    class_counts = data["label"].value_counts()
    data = data[data["label"].isin(class_counts[class_counts >= 2].index)]

    if data["label"].nunique() < 2:
        st.error("Ошибка: В датасете только один класс.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data["code"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, ML_MODEL_FILE)
    joblib.dump(vectorizer, VECTOR_FILE)

    y_pred = model.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    json.dump({"precision": precision, "recall": recall, "f1_score": f1}, open(METRICS_FILE, "w"))

    with open(TRAINING_LOG_FILE, "a") as log:
        log.write(f"Обучение: precision={precision}, recall={recall}, f1={f1}\n")

    st.success("Модель обучена!")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")

# === Безопасная загрузка модели ===
def load_ml_model():
    if not os.path.exists(ML_MODEL_FILE) or not os.path.exists(VECTOR_FILE):
        st.error("Файл модели или векторизатора не найден. Сначала обучите модель.")
        return None, None
    try:
        model = joblib.load(ML_MODEL_FILE)
        vectorizer = joblib.load(VECTOR_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

# === Анализ кода с моделью ===
def analyze_code_with_ml(code_snippet):
    model, vectorizer = load_ml_model()
    if model is None or vectorizer is None:
        return "Модель не загружена. Сначала обучите модель."
    try:
        vectorized_code = vectorizer.transform([code_snippet])
        return "Обнаружена уязвимость" if model.predict(vectorized_code)[0] == 1 else "Код безопасен"
    except Exception as e:
        return f"Ошибка при анализе: {e}"

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", [
        "Администрирование",
        "Обучение",
        "Эксплуатация",
        "Метрики модели"
    ])

    if menu == "Администрирование":
        st.subheader("Управление пользователями")
        choice = st.radio("Выберите действие", ["Вход", "Регистрация"])
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        if choice == "Регистрация" and st.button("Зарегистрироваться"):
            st.success(register_user(username, password))
        if choice == "Вход" and st.button("Войти"):
            st.success("Успешный вход") if login_user(username, password) else st.error("Неверные данные")

    elif menu == "Обучение":
        st.subheader("Обучение модели")
        if st.button("Обучить с нуля"):
            train_ml_model()

    elif menu == "Эксплуатация":
        st.subheader("Модуль эксплуатации")

        if "uploaded_code" not in st.session_state:
            st.session_state.uploaded_code = None

        uploaded_file = st.file_uploader("Загрузите файл кода (.py, .txt)", type=["py", "txt"])
        if uploaded_file:
            try:
                code = uploaded_file.read().decode("utf-8")
                st.session_state.uploaded_code = code
                st.success("Файл загружен. Вы можете выполнить анализ.")
            except UnicodeDecodeError:
                st.error("Ошибка: файл не в текстовом формате (UTF-8). Загрузите .py или .txt файл.")

        if st.session_state.uploaded_code:
            result = analyze_code_with_ml(st.session_state.uploaded_code)
            st.write("**Результат анализа:**")
            st.info(result)

            if st.button("Загрузить следующий файл"):
                st.session_state.uploaded_code = None
                st.experimental_rerun()

    elif menu == "Метрики модели":
        st.subheader("Метрики модели")
        if os.path.exists(METRICS_FILE):
            metrics = json.load(open(METRICS_FILE))
            st.write(f"**Precision:** {metrics['precision']:.4f}")
            st.write(f"**Recall:** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.info("Метрики обучения не найдены.")

if __name__ == "__main__":
    main()


