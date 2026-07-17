FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r ./service/requirements.txt

COPY . ./service/

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
