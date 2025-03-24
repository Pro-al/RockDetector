import streamlit as st

st.title("Автоматическое обнаружение уязвимостей")
st.write("Введите исходный код для анализа:")

code = st.text_area("Исходный код")
if st.button("Анализировать"):
    st.write("Результаты анализа: ...")
