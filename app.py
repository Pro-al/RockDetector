import streamlit as st
import pandas as pd
import os
import json
import joblib
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# === Пути к файлам ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"
RETRAIN_METRICS_FILE = "retrain_metrics.json"
FSTEC_DB_FILE = "fstec_db.json"

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
def train_model(save_to=METRICS_FILE):
    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        st.error("Файл с датасетом не найден.")
        return None

    class_counts = data["label"].value_counts()
    data = data[data["label"].isin(class_counts[class_counts >= 2].index)]

    if data["label"].nunique() < 2:
        st.error("Недостаточно различных классов в данных.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        data["code"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_vec, y_train)

    joblib.dump(model, ML_MODEL_FILE)
    joblib.dump(vectorizer, VECTOR_FILE)

    y_pred = model.predict(X_test_vec)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    json.dump({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "timestamp": datetime.now().isoformat()
    }, open(save_to, "w"))

    return precision, recall, f1

# === Загрузка модели ===
def load_model():
    if not os.path.exists(ML_MODEL_FILE) or not os.path.exists(VECTOR_FILE):
        return None, None
    try:
        model = joblib.load(ML_MODEL_FILE)
        vectorizer = joblib.load(VECTOR_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

# === Анализ кода ===
def analyze_code(code_snippet):
    model, vectorizer = load_model()
    if model is None:
        return "Модель не загружена. Пожалуйста, выполните обучение."
    vectorized = vectorizer.transform([code_snippet])
    result = model.predict(vectorized)[0]
    return "Обнаружена уязвимость" if result == 1 else "Код безопасен"

# === Сравнение с базой ФСТЭК ===
def load_fstec_db():
    if not os.path.exists(FSTEC_DB_FILE):
        return []
    try:
        return json.load(open(FSTEC_DB_FILE, "r", encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("Ошибка чтения базы ФСТЭК. Проверьте формат JSON.")
        return []

def check_fstec(code_snippet):
    db = load_fstec_db()
    hash_value = hashlib.sha256(code_snippet.encode()).hexdigest()
    for item in db:
        if item.get("hash") == hash_value:
            return {
                "description": item.get("description", "нет описания"),
                "CVE": item.get("CVE", "не указано"),
                "severity": item.get("severity", "не указано")
            }
    return None

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Меню", [
        "Администрирование",
        "Обучение",
        "Дообучение",
        "Эксплуатация",
        "Анализ кода",
        "Метрики"
    ])

    if menu == "Администрирование":
        st.subheader("Управление пользователями")
        action = st.radio("Выберите действие", ["Вход", "Регистрация"])
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        if action == "Регистрация" and st.button("Зарегистрироваться"):
            st.success(register_user(username, password))
        if action == "Вход" and st.button("Войти"):
            if login_user(username, password):
                st.success("Успешный вход")
            else:
                st.error("Неверные данные")

    elif menu == "Обучение":
        st.subheader("Обучение модели")
        if st.button("Обучить"):
            result = train_model(METRICS_FILE)
            if result:
                st.success("Обучение завершено")
                st.write(f"**Precision:** {result[0]:.4f}")
                st.write(f"**Recall:** {result[1]:.4f}")
                st.write(f"**F1-score:** {result[2]:.4f}")

    elif menu == "Дообучение":
        st.subheader("Дообучение модели")
        if st.button("Дообучить"):
            result = train_model(RETRAIN_METRICS_FILE)
            if result:
                st.success("Дообучение завершено")
                st.write(f"**Precision:** {result[0]:.4f}")
                st.write(f"**Recall:** {result[1]:.4f}")
                st.write(f"**F1-score:** {result[2]:.4f}")

    elif menu == "Эксплуатация":
        st.subheader("Проверка кода")
        uploaded = st.file_uploader("Загрузите файл кода (.py, .txt)", type=["py", "txt"])
        if uploaded:
            try:
                content = uploaded.read().decode("utf-8")
                result = analyze_code(content)
                st.write("**Результат анализа:**")
                st.info(result)

                match = check_fstec(content)
                if match:
                    st.write("**Найдено совпадение в базе ФСТЭК:**")
                    st.write(f"- **Описание:** {match['description']}")
                    st.write(f"- **CVE:** {match['CVE']}")
                    st.write(f"- **Серьёзность:** {match['severity']}")
                else:
                    st.write("Совпадений в базе ФСТЭК не найдено.")
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

    elif menu == "Анализ кода":
        st.subheader("Ввод кода вручную")
        code_input = st.text_area("Введите фрагмент кода")
        if st.button("Анализировать"):
            result = analyze_code(code_input)
            st.info(result)

            match = check_fstec(code_input)
            if match:
                st.write("**Совпадение в базе ФСТЭК:**")
                st.write(f"- **Описание:** {match['description']}")
                st.write(f"- **CVE:** {match['CVE']}")
                st.write(f"- **Серьёзность:** {match['severity']}")
            else:
                st.write("Совпадений в базе ФСТЭК не найдено.")

    elif menu == "Метрики":
        st.subheader("Метрики обучения")
        if os.path.exists(METRICS_FILE):
            metrics = json.load(open(METRICS_FILE))
            st.write(f"**Точность (Precision):** {metrics['precision']:.4f}")
            st.write(f"**Полнота (Recall):** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.info("Модель ещё не обучена.")

        st.subheader("Метрики после дообучения")
        if os.path.exists(RETRAIN_METRICS_FILE):
            metrics = json.load(open(RETRAIN_METRICS_FILE))
            st.write(f"**Точность (Precision):** {metrics['precision']:.4f}")
            st.write(f"**Полнота (Recall):** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.info("Дообучение ещё не производилось.")

if __name__ == "__main__":
    main()
