import streamlit as st
import pandas as pd
import json
import os
import hashlib
import requests

# Попытка импорта необходимых библиотек с обработкой ошибок
try:
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
    import joblib
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Убедитесь, что установлены все необходимые библиотеки.")
    st.write("Чтобы установить необходимые библиотеки, выполните команду:")
    st.code("pip install scikit-learn joblib matplotlib")
    st.stop()  # Останавливаем выполнение, если не удалось импортировать библиотеки

# === Глобальные переменные ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
FSTEC_DB_FILE = "fstec_db.json"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"

# === Функции работы с пользователями ===
def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r", encoding="utf-8") as file:
        return json.load(file)

def save_users(users):
    with open(USER_DB, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=4)

def register_user(username, password):
    users = load_users()
    if username in users:
        return "Пользователь уже существует"
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    save_users(users)
    return "Регистрация успешна"

def login_user(username, password):
    users = load_users()
    if username in users and users[username] == hashlib.sha256(password.encode()).hexdigest():
        return True
    return False

# === Обучение модели ===
def train_ml_model():
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

    # Оценка модели
    y_pred = model.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Сохранение метрик
    metrics = {"precision": precision, "recall": recall, "f1_score": f1}
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)

    # Визуализация Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    plt.figure()
    plt.plot(recall_vals, precision_vals, marker='.', label='PR-кривая')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    st.pyplot(plt)

    st.success("Модель обучена и сохранена!")

# === Загрузка обученной модели ===
def load_ml_model():
    try:
        model = joblib.load(ML_MODEL_FILE)
        vectorizer = joblib.load(VECTOR_FILE)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Обученная модель не найдена.")
        return None, None

# === Анализ кода с помощью ML ===
def analyze_code_with_ml(code_snippet):
    model, vectorizer = load_ml_model()
    if model is None:
        return "Ошибка загрузки модели"
    
    vectorized_code = vectorizer.transform([code_snippet])
    prediction = model.predict(vectorized_code)
    return "Обнаружена уязвимость" if prediction[0] == 1 else "Код безопасен"

# === Работа с БДУ ФСТЭК ===
def load_fstec_db():
    if not os.path.exists(FSTEC_DB_FILE):
        return []
    with open(FSTEC_DB_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def compare_with_fstec(code_snippet):
    fstec_db = load_fstec_db()
    code_hash = hashlib.sha256(code_snippet.encode()).hexdigest()
    for vuln in fstec_db:
        if vuln["hash"] == code_hash:
            return f"Совпадение с БДУ ФСТЭК: {vuln['description']}"
    return "Совпадений с БДУ ФСТЭК не найдено"

def update_fstec_db():
    st.subheader("Обновление базы данных уязвимостей ФСТЭК")
    api_url = st.text_input("Введите API-адрес для загрузки базы ФСТЭК")

    if st.button("Обновить базу"):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                new_db = response.json()
                for vuln in new_db:
                    vuln["hash"] = hashlib.sha256(vuln["pattern"].encode()).hexdigest()
                with open(FSTEC_DB_FILE, "w", encoding="utf-8") as file:
                    json.dump(new_db, file, indent=4)
                st.success("База данных ФСТЭК обновлена!")
            else:
                st.error("Ошибка загрузки базы")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", ["Администрирование", "Обучение", "Эксплуатация", "Анализ кода", "Обновление ФСТЭК"])

    # === Администрирование ===
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

    # === Эксплуатация (анализ кода) ===
    elif menu == "Эксплуатация":
        st.subheader("Анализ загруженного кода")
        uploaded_file = st.file_uploader("Загрузите файл кода", type=["py", "java", "js", "c", "cpp"])
        
        if uploaded_file:
            code_snippet = uploaded_file.read().decode("utf-8")
            result = analyze_code_with_ml(code_snippet)
            st.write("Результат анализа:", result)

    # === Анализ с БДУ ФСТЭК ===
    elif menu == "Анализ кода":
        st.subheader("Анализ кода с БДУ ФСТЭК")
        code_input = st.text_area("Введите фрагмент кода для анализа")

        if st.button("Анализировать"):
            result = compare_with_fstec(code_input)
            st.write("Результат:", result)

    # === Обновление БДУ ФСТЭК ===
    elif menu == "Обновление ФСТЭК":
        update_fstec_db()

if __name__ == "__main__":
    main()
