import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("📊 Machine Learning Predictor")
st.markdown("Upload your CSV file and get predictions from the deployed model.")

# Backend URL input
backend_url = st.text_input("Enter FastAPI URL", "https://your-fastapi-url.onrender.com/predict")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Validate and preview uploaded data
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Invalid CSV file: {str(e)}")
        st.stop()

    # Button to send file to FastAPI backend
    if st.button("🔮 Get Predictions"):
        with st.spinner("Sending file to model for prediction..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(backend_url, files=files)

                if response.status_code == 200:
                    st.success("✅ Prediction successful!")

                    # Convert returned HTML table back to DataFrame
                    if "text/html" in response.headers.get("Content-Type", ""):
                        result_html = response.text
                        df_result = pd.read_html(result_html, flavor='bs4')[0]

                        st.subheader("📈 Prediction Results")
                        st.dataframe(df_result)

                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")
                    else:
                        st.error(f"❌ Unexpected response format: {response.text}")
                else:
                    st.error(f"❌ Error: {response.text}")

            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")