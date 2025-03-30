import streamlit as st
import pandas as pd
import json
import os
import hashlib
import requests

try:
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
    import joblib
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Установите библиотеки: `pip install scikit-learn joblib matplotlib`")
    st.stop()

# === Файлы ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
FSTEC_DB_FILE = "fstec_db.json"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"

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

    # Удаляем редкие классы (< 2 записей)
    class_counts = data["label"].value_counts()
    data = data[data["label"].isin(class_counts[class_counts >= 2].index)]

    if data["label"].nunique() < 2:
        st.error("Ошибка: В датасете только один класс. Добавьте данные.")
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

    # Оценка модели
    y_pred = model.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    json.dump({"precision": precision, "recall": recall, "f1_score": f1}, open(METRICS_FILE, "w"))

    # Визуализация Precision-Recall (если есть 2 класса)
    if model.n_classes_ > 1:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1])
        plt.figure()
        plt.plot(recall_vals, precision_vals, marker='.', label='PR-кривая')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Невозможно построить PR-кривую (1 класс).")

    st.success("Модель обучена и сохранена!")

# === Загрузка модели ===
def load_ml_model():
    try:
        return joblib.load(ML_MODEL_FILE), joblib.load(VECTOR_FILE)
    except FileNotFoundError:
        st.error("Обученная модель не найдена.")
        return None, None

# === Анализ кода ===
def analyze_code_with_ml(code_snippet):
    model, vectorizer = load_ml_model()
    if model is None:
        return "Ошибка загрузки модели"
    
    vectorized_code = vectorizer.transform([code_snippet])
    return "Обнаружена уязвимость" if model.predict(vectorized_code)[0] == 1 else "Код безопасен"

# === Работа с БДУ ФСТЭК ===
def load_fstec_db():
    return json.load(open(FSTEC_DB_FILE, "r", encoding="utf-8")) if os.path.exists(FSTEC_DB_FILE) else []

def compare_with_fstec(code_snippet):
    fstec_db = load_fstec_db()
    code_hash = hashlib.sha256(code_snippet.encode()).hexdigest()
    for vuln in fstec_db:
        if vuln.get("hash") == code_hash:
            return f"Совпадение с БДУ ФСТЭК: {vuln['description']}"
    return "Совпадений с БДУ ФСТЭК не найдено"

def update_fstec_db():
    st.subheader("Обновление БДУ ФСТЭК")
    api_url = st.text_input("Введите API-адрес")

    if st.button("Обновить базу"):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                new_db = response.json()
                for vuln in new_db:
                    vuln["hash"] = hashlib.sha256(vuln["pattern"].encode()).hexdigest()
                json.dump(new_db, open(FSTEC_DB_FILE, "w", encoding="utf-8"), indent=4)
                st.success("База ФСТЭК обновлена!")
            else:
                st.error("Ошибка загрузки базы")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", ["Администрирование", "Обучение", "Эксплуатация", "Анализ кода", "Обновление ФСТЭК"])

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
        train_ml_model()

    elif menu == "Эксплуатация":
        uploaded_file = st.file_uploader("Загрузите файл кода")
        if uploaded_file:
            st.write("Результат анализа:", analyze_code_with_ml(uploaded_file.read().decode("utf-8")))

    elif menu == "Анализ кода":
        code_input = st.text_area("Введите код")
        if st.button("Анализировать"):
            st.write("Результат:", compare_with_fstec(code_input))

    elif menu == "Обновление ФСТЭК":
        update_fstec_db()

if __name__ == "__main__":
    main()

     
