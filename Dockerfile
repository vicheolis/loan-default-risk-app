# Base image: a small, official Python environment
FROM python:3.11-slim

# Work inside /app inside the container
WORKDIR /app

# Copy dependency list first (Docker caches this layer, so rebuilds are faster
# when only app.py changes and requirements.txt doesn't)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual app code and the trained model
COPY app.py .
COPY selected_loan_default_model.joblib .

# Streamlit's default port
EXPOSE 8501

# Command that runs when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
