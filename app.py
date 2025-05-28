import streamlit as st
import pandas as pd
import json
import os
import hashlib
import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# === Константы ===
USER_DB = "users.json"
ML_MODEL_FILE = "ml_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
DATASET_FILE = "vulnerability_dataset.csv"
METRICS_FILE = "metrics.json"
TRAINING_LOG_FILE = "training_log.txt"
FSTEC_DB_FILE = "fstec_db.json"
MITRE_DB_FILE = "mitre_db.json"

# === Пользователи ===
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
        st.error("Недостаточно классов.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data["code"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
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


# === Загрузка модели (безопасно) ===
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


# === Анализ кода ===
def analyze_code_with_ml(code_snippet):
    model, vectorizer = load_ml_model()
    if model is None or vectorizer is None:
        return "Модель не загружена. Обучите модель."
    try:
        vectorized = vectorizer.transform([code_snippet])
        return "Обнаружена уязвимость" if model.predict(vectorized)[0] == 1 else "Код безопасен"
    except Exception as e:
        return f"Ошибка при анализе: {e}"


# === Работа с ФСТЭК ===
def load_fstec_db():
    if os.path.exists(FSTEC_DB_FILE):
        with open(FSTEC_DB_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                st.error("Ошибка чтения базы ФСТЭК.")
    return []

def update_fstec_db():
    st.subheader("Обновление БДУ ФСТЭК")
    api_url = st.text_input("Введите API-адрес для обновления")
    if st.button("Обновить базу"):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                new_db = response.json()
                if isinstance(new_db, list):
                    for vuln in new_db:
                        if "pattern" in vuln:
                            vuln["hash"] = hashlib.sha256(vuln["pattern"].encode()).hexdigest()
                    with open(FSTEC_DB_FILE, "w", encoding="utf-8") as f:
                        json.dump(new_db, f, indent=4)
                    st.success("База ФСТЭК успешно обновлена.")
                else:
                    st.error("Ожидался список уязвимостей.")
            else:
                st.error(f"Ошибка от сервера: код {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка запроса: {e}")

def add_mitre_entry():
    st.subheader("Добавить MITRE-уязвимость вручную")
    mitre_id = st.text_input("MITRE ID (например, CWE-89)")
    name = st.text_input("Название уязвимости")
    description = st.text_area("Описание")
    severity = st.selectbox("Уровень критичности", ["Low", "Medium", "High", "Critical"])

    if st.button("Добавить в базу MITRE"):
        entry = {
            "id": mitre_id,
            "name": name,
            "description": description,
            "severity": severity
        }
        mitre_db = []
        if os.path.exists(MITRE_DB_FILE):
            try:
                mitre_db = json.load(open(MITRE_DB_FILE, "r", encoding="utf-8"))
            except:
                pass
        mitre_db.append(entry)
        with open(MITRE_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(mitre_db, f, indent=4, ensure_ascii=False)
        st.success("Запись успешно добавлена в базу MITRE.")


# === Интерфейс Streamlit ===
def main():
    st.title("Система анализа уязвимостей")

    menu = st.sidebar.radio("Выберите модуль", [
        "Администрирование", 
        "Обучение", 
        "Эксплуатация", 
        "Обновление ФСТЭК / MITRE", 
        "Метрики"
    ])

    if menu == "Администрирование":
        st.subheader("Управление пользователями")
        action = st.radio("Выберите действие", ["Вход", "Регистрация"])
        login = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        if action == "Регистрация" and st.button("Зарегистрироваться"):
            st.success(register_user(login, password))
        if action == "Вход" and st.button("Войти"):
            if login_user(login, password):
                st.success("Успешный вход")
            else:
                st.error("Неверный логин или пароль")

    elif menu == "Обучение":
        st.subheader("Модуль обучения")
        if st.button("Обучить с нуля"):
            train_ml_model()

    elif menu == "Эксплуатация":
        st.subheader("Модуль эксплуатации")
        if "uploaded_code" not in st.session_state:
            st.session_state.uploaded_code = None

        uploaded_file = st.file_uploader("Загрузите .py или .txt файл", type=["py", "txt"])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                st.session_state.uploaded_code = content
                st.success("Файл загружен.")
            except UnicodeDecodeError:
                st.error("Ошибка: файл должен быть в кодировке UTF-8.")

        if st.session_state.uploaded_code:
            result = analyze_code_with_ml(st.session_state.uploaded_code)
            st.write("**Результат анализа:**")
            st.info(result)

            if st.button("Анализировать другой файл"):
                st.session_state.uploaded_code = None
                st.experimental_rerun()

    elif menu == "Обновление ФСТЭК / MITRE":
        update_fstec_db()
        st.markdown("---")
        add_mitre_entry()

    elif menu == "Метрики":
        st.subheader("Метрики модели")
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
            st.write(f"**Precision:** {metrics['precision']:.4f}")
            st.write(f"**Recall:** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.info("Метрики ещё не сохранены. Обучите модель.")

if __name__ == "__main__":
    main()
