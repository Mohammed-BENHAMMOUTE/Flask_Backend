FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir flask-cors
RUN python -c "from flask_cors import CORS; print('Flask-CORS imported successfully')"

COPY . .

CMD ["python", "app.py"]