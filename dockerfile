FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose port if you later add a Flask/Streamlit app
EXPOSE 5000

CMD ["python", "main.py"]
