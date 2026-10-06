import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Student Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Student Academic Performance Assistant")
st.write("Predict your academic performance and chat with your AI assistant.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Choose an option",
    ["📊 Performance Prediction", "🤖 AI Chatbot"]
)

# =========================================================
# PERFORMANCE PREDICTION
# =========================================================

if page == "📊 Performance Prediction":

    st.header("📊 Student Academic Performance Prediction")

    st.write(
        "Enter your academic details below to predict your expected score."
    )

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75
        )

        internal_marks = st.slider(
            "Internal Marks (out of 40)",
            min_value=0,
            max_value=40,
            value=25
        )

    with col2:
        assignment_marks = st.slider(
            "Assignment Marks (out of 10)",
            min_value=0,
            max_value=10,
            value=7
        )

        study_hours = st.slider(
            "Study Hours per Day",
            min_value=0,
            max_value=12,
            value=3
        )

    # ---------------- SAMPLE TRAINING DATA ----------------
    data = pd.DataFrame({
        "attendance": [
            50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
            60, 70, 80, 90, 45
        ],
        "internal_marks": [
            15, 18, 20, 22, 25, 27, 30, 32, 35, 38,
            21, 26, 31, 36, 12
        ],
        "assignment_marks": [
            4, 5, 5, 6, 6, 7, 8, 8, 9, 10,
            5, 7, 8, 9, 3
        ],
        "study_hours": [
            1, 1, 2, 2, 3, 3, 4, 5, 5, 6,
            2, 3, 4, 5, 1
        ],
        "score": [
            45, 48, 52, 57, 62, 67, 72, 78, 85, 92,
            54, 65, 75, 88, 40
        ]
    })

    X = data[
        [
            "attendance",
            "internal_marks",
            "assignment_marks",
            "study_hours"
        ]
    ]

    y = data["score"]

    # ---------------- TRAIN MODEL ----------------
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    # ---------------- PREDICTION ----------------
    if st.button("🔮 Predict Performance"):

        student_data = pd.DataFrame({
            "attendance": [attendance],
            "internal_marks": [internal_marks],
            "assignment_marks": [assignment_marks],
            "study_hours": [study_hours]
        })

        prediction = model.predict(student_data)[0]

        prediction = max(0, min(100, prediction))

        st.subheader("📈 Prediction Result")

        st.metric(
            "Predicted Academic Score",
            f"{prediction:.2f}%"
        )

        if prediction >= 75:
            st.success(
                "🌟 Excellent performance! Keep maintaining your study routine."
            )

        elif prediction >= 60:
            st.info(
                "👍 Good performance! A little more effort can improve your score."
            )

        elif prediction >= 40:
            st.warning(
                "⚠️ Average performance. Try to improve attendance and study hours."
            )

        else:
            st.error(
                "❗ Low predicted performance. You should focus more on your studies."
            )

        # ---------------- PERFORMANCE CHART ----------------

        chart_data = pd.DataFrame({
            "Category": [
                "Attendance",
                "Internal Marks",
                "Assignment Marks",
                "Study Hours"
            ],
            "Value": [
                attendance,
                internal_marks,
                assignment_marks,
                study_hours
            ]
        })

        st.subheader("📊 Student Details")

        st.bar_chart(
            chart_data.set_index("Category")
        )


# =========================================================
# AI CHATBOT
# =========================================================

elif page == "🤖 AI Chatbot":

    st.header("🤖 Student AI Chatbot")

    st.write(
        "Ask questions about studies, exams, attendance, "
        "assignments and academic performance."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input(
        "Ask your academic question..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):

            with st.spinner("AI is thinking..."):

                try:

                    response = requests.post(
                        "http://localhost:11434/api/chat",

                        json={
                            "model": "llama3.2",

                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an AI Student Academic "
                                        "Assistant. Help students with "
                                        "academic questions, study plans, "
                                        "exam preparation, attendance, "
                                        "assignments and performance. "
                                        "Give simple and clear answers."
                                    )
                                },
                                *st.session_state.messages
                            ],

                            "stream": False
                        },

                        timeout=120
                    )

                    response.raise_for_status()

                    answer = response.json()["message"]["content"]

                    st.write(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except requests.RequestException:

                    st.error(
                        "❌ Could not connect to Ollama. "
                        "Please make sure Ollama is running."
                    )