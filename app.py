import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from groq import Groq
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI-Based NIDS", layout="wide")

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**  
- Machine Learning based Intrusion Detection  
- Explainable AI using Groq LLM  
""")

DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# ---------------- DATA LOADING ----------------
@st.cache_data(show_spinner=False)
def load_data(path: str, n_rows: int = 15000) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=n_rows)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ---------------- MODEL TRAINING ----------------
def train_model(df: pd.DataFrame):
    features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Fwd Packet Length Max",
        "Flow IAT Mean", "Flow IAT Std", "Flow Packets/s",
    ]
    target = "Label"

    # Safety check: ensure required columns exist
    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X_test, y_test, features

# ---------------- MAIN APP ----------------
try:
    df = load_data(DATA_FILE)
    st.sidebar.success(f"Dataset Loaded: {len(df)} rows")
except FileNotFoundError:
    st.error(f"Dataset file not found: {DATA_FILE}")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if st.sidebar.button("Train Model"):
    with st.spinner("Training Model..."):
        try:
            model, acc, X_test, y_test, features = train_model(df)
            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["features"] = features
            st.sidebar.success(f"Accuracy: {acc:.2%}")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# ---------------- DASHBOARD ----------------
if "model" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Packet Simulation")

        if st.button("Capture Random Packet"):
            if len(st.session_state["X_test"]) == 0:
                st.warning("No test data available.")
            else:
                idx = np.random.randint(0, len(st.session_state["X_test"]))
                st.session_state["packet"] = st.session_state["X_test"].iloc[idx]
                st.session_state["actual"] = st.session_state["y_test"].iloc[idx]

        if "packet" in st.session_state:
            st.write("Sample Packet Features:")
            st.dataframe(st.session_state["packet"].to_frame(name="Value"))

    with col2:
        if "packet" in st.session_state:
            packet = st.session_state["packet"]
            model = st.session_state["model"]

            packet_array = packet.to_numpy().reshape(1, -1)
            prediction = model.predict(packet_array)[0]
            proba = model.predict_proba(packet_array).max()

            if prediction == "BENIGN":
                st.success(f"SAFE TRAFFIC (Confidence: {proba:.2%})")
            else:
                st.error(f"ATTACK DETECTED: {prediction} (Confidence: {proba:.2%})")

            st.caption(f"Actual Label: {st.session_state.get('actual', 'N/A')}")

            # -------- GROQ EXPLANATION --------
            st.markdown("### AI Explanation")
            if st.button("Explain Packet"):
                if not groq_api_key:
                    st.warning("Enter Groq API Key in sidebar")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)
                        prompt = f"""
                        Explain why this packet is classified as {prediction}.
                        Packet Features:
                        {packet.to_string()}
                        Keep explanation simple for students.
                        """
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.5,
                        )
                        st.info(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Groq explanation failed: {e}")

    # -------- FEATURE IMPORTANCE --------
    st.markdown("---")
    st.subheader("Feature Importance")

    importances = st.session_state["model"].feature_importances_
    fi_df = (
        pd.DataFrame({
            "Feature": st.session_state["features"],
            "Importance": importances,
        })
        .sort_values(by="Importance", ascending=False)
    )

    fig, ax = plt.subplots()
    sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["model"].predict(st.session_state["X_test"])
    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

else:
    st.info("Train the model to start detection.")
