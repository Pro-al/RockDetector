import streamlit as st
import pandas as pd
import json
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import hashlib

# === Глобальные переменные ===
USER_DB = "users.json"  # Файл базы пользователей
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
FSTEC_DB_FILE = "fstec_db.json"
DATASET_FILE = "vulnerability_dataset.csv"

# === Модуль 1: Администрирование (управление пользователями) ===
def load_users():
    """Загрузка базы пользователей"""
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r", encoding="utf-8") as file:
        return json.load(file)

def save_users(users):
    """Сохранение базы пользователей"""
    with open(USER_DB, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)

def register_user(username, password):
    """Регистрация нового пользователя"""
    users = load_users()
    if username in users:
        return "Пользователь уже существует"
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    save_users(users)
    return "Регистрация успешна"

def login_user(username, password):
    """Авторизация пользователя"""
    users = load_users()
    if username in users and users[username] == hashlib.sha256(password.encode()).hexdigest():
        return True
    return False

# === Модуль 2: Обучение модели машинного обучения ===
def train_ml_model():
    """Обучение модели для анализа уязвимостей"""
    st.subheader("Обучение модели")
    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        st.error("Файл датасета не найден.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(data["code"], data["label"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, ML_MODEL_FILE)
    joblib.dump(vectorizer, VECTOR_FILE)

    st.success("Модель обучена и сохранена!")

# === Модуль 3: Эксплуатация (анализ загружаемого кода) ===
def load_ml_model():
    """Загрузка обученной модели"""
    try:
        model = joblib.load(ML_MODEL_FILE)
        vectorizer = joblib.load(VECTOR_FILE)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Обученная модель не найдена.")
        return None, None

def analyze_code_with_ml(code_snippet):
    """Анализ кода с помощью обученной модели"""
    model, vectorizer = load_ml_model()
    if model is None:
        return "Ошибка загрузки модели"
    
    vectorized_code = vectorizer.transform([code_snippet])
    prediction = model.predict(vectorized_code)
    return "Обнаружена уязвимость" if prediction[0] == 1 else "Код безопасен"

# === Модуль 4: Анализ исходного кода с БДУ ФСТЭК ===
def load_fstec_db():
    """Загрузка базы данных уязвимостей ФСТЭК"""
    if not os.path.exists(FSTEC_DB_FILE):
        return []
    with open(FSTEC_DB_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def compare_with_fstec(code_snippet):
    """Сравнение кода с известными уязвимостями из БДУ ФСТЭК"""
    fstec_db = load_fstec_db()
    for vuln in fstec_db:
        if vuln["pattern"] in code_snippet:
            return f"Совпадение с БДУ ФСТЭК: {vuln['description']}"
    return "Совпадений с БДУ ФСТЭК не найдено"

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", ["Администрирование", "Обучение", "Эксплуатация", "Анализ кода"])

    # === Администрирование (вход в систему) ===
    if menu == "Администрирование":
        st.subheader("Управление пользователями")
        choice = st.radio("Выберите действие", ["Вход", "Регистрация"])
        
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")

        if choice == "Регистрация":
            if st.button("Зарегистрироваться"):
                result = register_user(username, password)
                st.success(result)

        elif choice == "Вход":
            if st.button("Войти"):
                if login_user(username, password):
                    st.success("Успешный вход")
                else:
                    st.error("Неверные учетные данные")

    # === Обучение модели ===
    elif menu == "Обучение":
        train_ml_model()

    # === Эксплуатация (анализ загружаемого кода) ===
    elif menu == "Эксплуатация":
        st.subheader("Анализ загруженного кода")
        uploaded_file = st.file_uploader("Загрузите файл кода", type=["py", "java", "js", "c", "cpp"])
        
        if uploaded_file:
            code_snippet = uploaded_file.read().decode("utf-8")
            result = analyze_code_with_ml(code_snippet)
            st.write("Результат анализа:", result)

    # === Анализ исходного кода с БДУ ФСТЭК ===
    elif menu == "Анализ кода":
        st.subheader("Анализ кода с БДУ ФСТЭК")
        code_input = st.text_area("Введите фрагмент кода для анализа")

        if st.button("Анализировать"):
            result = compare_with_fstec(code_input)
            st.write("Результат:", result)

if __name__ == "__main__":
    main()

