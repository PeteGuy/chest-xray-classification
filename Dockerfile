FROM python:3.10.12-slim

WORKDIR /app 

COPY . .


RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install typing_extensions==4.15.0
RUN pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


CMD ["python","-m","app.main"]
