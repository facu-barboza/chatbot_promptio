FROM python:3.10-slim

WORKDIR /app

# Copy archives to container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY chatbot_pipeline.py chatbot_pipeline.py
COPY AIEngineer.pdf AIEngineer.pdf 

# Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Flask port
EXPOSE 5000

CMD ["python", "app.py"]

