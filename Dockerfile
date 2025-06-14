FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["bash", "-c", "python cargapsql.py && python app.py"]

