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
FSTEC_DB_FILE = "fstec_db.json"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"
TRAINING_LOG_FILE = "training_log.txt"

# === Функции работы с пользователями ===
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

# === Функция обучения модели ===
def train_ml_model():
    st.subheader("Обучение модели")
    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        st.error("Файл датасета не найден.")
        return

    # Удаление редких классов
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
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Сохранение метрик в файл
    json.dump({"precision": precision, "recall": recall, "f1_score": f1}, open(METRICS_FILE, "w"))

    # Логирование процесса обучения
    with open(TRAINING_LOG_FILE, "a") as log:
        log.write(f"Обучение: precision={precision}, recall={recall}, f1={f1}\n")
    
    # Отображение метрик
    st.success("Модель обучена и сохранена!")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")

    # Визуализация графиков
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    st.pyplot(plt)

# === Загрузка модели ===
def load_ml_model():
    try:
        return joblib.load(ML_MODEL_FILE), joblib.load(VECTOR_FILE)
    except FileNotFoundError:
        st.error("Обученная модель не найдена.")
        return None, None

# === Анализ кода через ML ===
def analyze_code_with_ml(code_snippet):
    model, vectorizer = load_ml_model()
    if model is None:
        return "Ошибка загрузки модели"
    
    vectorized_code = vectorizer.transform([code_snippet])
    return "Обнаружена уязвимость" if model.predict(vectorized_code)[0] == 1 else "Код безопасен"

# === Работа с БДУ ФСТЭК ===
def load_fstec_db():
    if os.path.exists(FSTEC_DB_FILE):
        with open(FSTEC_DB_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                st.error("Ошибка в файле базы ФСТЭК. Проверьте формат JSON.")
                return []
    return []

def compare_with_fstec(code_snippet):
    fstec_db = load_fstec_db()
    code_hash = hashlib.sha256(code_snippet.encode()).hexdigest()

    for vuln in fstec_db:
        if vuln.get("hash") == code_hash:
            return f"**Совпадение с БДУ ФСТЭК найдено:**\n\n" \
                   f"**Уязвимость:** {vuln['description']}\n" \
                   f"**CVE:** {vuln['CVE']}\n" \
                   f"**Серьезность:** {vuln['severity']}"

    return "Совпадений с БДУ ФСТЭК не найдено"

def update_fstec_db():
    st.subheader("Обновление БДУ ФСТЭК")
    api_url = st.text_input("Введите API-адрес")

    if st.button("Обновить базу"):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                new_db = response.json()
                if isinstance(new_db, list):  # Проверка, что API вернул список
                    for vuln in new_db:
                        if "pattern" in vuln:
                            vuln["hash"] = hashlib.sha256(vuln["pattern"].encode()).hexdigest()
                    json.dump(new_db, open(FSTEC_DB_FILE, "w", encoding="utf-8"), indent=4)
                    st.success("База ФСТЭК обновлена!")
                else:
                    st.error("Ошибка: Ожидался список уязвимостей в формате JSON.")
            else:
                st.error(f"Ошибка: Сервер вернул код {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", ["Администрирование", "Обучение", "Эксплуатация", "Анализ кода", "Обновление ФСТЭК", "Метрики модели"])

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

    elif menu == "Метрики модели":
        st.subheader("Метрики модели")
        if os.path.exists(METRICS_FILE):
            metrics = json.load(open(METRICS_FILE, "r"))
            st.write(f"**Precision:** {metrics['precision']:.4f}")
            st.write(f"**Recall:** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.info("Модель ещё не обучена или метрики не сохранены.")

if __name__ == "__main__":
    main()
