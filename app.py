import streamlit as st
import pandas as pd
import os
import json
import joblib
import hashlib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

# === Пути к файлам ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"
RETRAIN_METRICS_FILE = "retrain_metrics.json"
FSTEC_DB_FILE = "fstec_db.json"
MITRE_DB_FILE = "mitre_db.json"

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
def train_model(save_to, plot_title):
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
    y_proba = model.predict_proba(X_test_vec)

    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    json.dump({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "timestamp": datetime.now().isoformat()
    }, open(save_to, "w"))

    # === Визуализация Precision-Recall
    try:
        pos_class_idx = list(model.classes_).index(1)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba[:, pos_class_idx])
        plt.figure(figsize=(6, 4))
        plt.plot(recall_vals, precision_vals, label="PR Curve")
        plt.title(f'Precision-Recall Curve ({plot_title})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        st.pyplot(plt)
    except:
        st.warning("Не удалось построить график PR. Вероятно, классы не бинарные.")

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
    if model is None or vectorizer is None:
        return None, "Модель не загружена."

    try:
        vectorized = vectorizer.transform([code_snippet])
        prediction = model.predict(vectorized)[0]
        return prediction, "Обнаружена уязвимость" if str(prediction) == "1" else "Код безопасен"
    except Exception as e:
        return None, f"Ошибка анализа: {e}"

# === Проверка баз (ФСТЭК, MITRE) ===
def check_vulnerability_db(code_snippet, db_file, label):
    if not os.path.exists(db_file):
        return None

    try:
        data = json.load(open(db_file, "r", encoding="utf-8"))
    except:
        return None

    code_hash = hashlib.sha256(code_snippet.encode()).hexdigest()
    for entry in data:
        if entry.get("hash") == code_hash:
            return {
                "label": label,
                "description": entry.get("description", ""),
                "CVE": entry.get("CVE", "не указано"),
                "severity": entry.get("severity", "не указано")
            }
    return None

# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", [
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
        if st.button("Обучить модель"):
            result = train_model(METRICS_FILE, "Обучение")
            if result:
                st.success("Обучение завершено")
                st.write(f"Precision: {result[0]:.4f}")
                st.write(f"Recall: {result[1]:.4f}")
                st.write(f"F1-score: {result[2]:.4f}")

    elif menu == "Дообучение":
        st.subheader("Дообучение модели")
        if st.button("Запустить дообучение"):
            result = train_model(RETRAIN_METRICS_FILE, "Дообучение")
            if result:
                st.success("Дообучение завершено")
                st.write(f"Precision: {result[0]:.4f}")
                st.write(f"Recall: {result[1]:.4f}")
                st.write(f"F1-score: {result[2]:.4f}")

    elif menu == "Эксплуатация":
        st.subheader("Анализ загруженного файла")
        uploaded_file = st.file_uploader(
            "Загрузите файл (.py, .txt, .csv, .html, .xss, .json, .php)", 
            type=["py", "txt", "csv", "html", "xss", "json", "php"]
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                try:
                    content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = file_bytes.decode("latin-1")

                if not content.strip():
                    st.warning("Файл пустой.")
                else:
                    st.success("Файл прочитан.")
                    label, result = analyze_code(content)
                    st.info(result)

                    for db_file, db_label in [(FSTEC_DB_FILE, "ФСТЭК"), (MITRE_DB_FILE, "MITRE")]:
                        match = check_vulnerability_db(content, db_file, db_label)
                        if match:
                            st.write(f"🔍 Совпадение в базе {match['label']}:")
                            st.write(f"- **Описание:** {match['description']}")
                            st.write(f"- **CVE:** {match['CVE']}")
                            st.write(f"- **Серьезность:** {match['severity']}")
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

    elif menu == "Анализ кода":
        st.subheader("Ручной анализ")
        code_input = st.text_area("Введите фрагмент кода")
        if st.button("Анализировать"):
            label, result = analyze_code(code_input)
            st.info(result)
            for db_file, db_label in [(FSTEC_DB_FILE, "ФСТЭК"), (MITRE_DB_FILE, "MITRE")]:
                match = check_vulnerability_db(code_input, db_file, db_label)
                if match:
                    st.write(f"🔍 Совпадение в базе {match['label']}:")
                    st.write(f"- **Описание:** {match['description']}")
                    st.write(f"- **CVE:** {match['CVE']}")
                    st.write(f"- **Серьезность:** {match['severity']}")

    elif menu == "Метрики":
        st.subheader("Метрики обучения")
        if os.path.exists(METRICS_FILE):
            m = json.load(open(METRICS_FILE))
            st.write(f"**Precision:** {m['precision']:.4f}")
            st.write(f"**Recall:** {m['recall']:.4f}")
            st.write(f"**F1-score:** {m['f1_score']:.4f}")
            st.write(f"_Дата: {m.get('timestamp', '')}_")
        else:
            st.info("Модель ещё не обучена.")

        st.subheader("Метрики дообучения")
        if os.path.exists(RETRAIN_METRICS_FILE):
            m = json.load(open(RETRAIN_METRICS_FILE))
            st.write(f"**Precision:** {m['precision']:.4f}")
            st.write(f"**Recall:** {m['recall']:.4f}")
            st.write(f"**F1-score:** {m['f1_score']:.4f}")
            st.write(f"_Дата: {m.get('timestamp', '')}_")
        else:
            st.info("Дообучение не выполнялось.")

if __name__ == "__main__":
    main()
