import json
import time
from kafka import KafkaProducer
from dotenv import load_dotenv
import os
import random

# Load environment variables
load_dotenv()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Sample data
sample_patients = [
    {"id": 1, "name": "John Doe", "details": "45-year-old male with hypertension"},
    {"id": 2, "name": "Jane Smith", "details": "32-year-old female with diabetes"},
    {"id": 3, "name": "Bob Johnson", "details": "58-year-old male with arthritis"}
]

sample_reports = [
    {"id": 1, "title": "Blood Test Results", "content": "Patient shows elevated cholesterol levels"},
    {"id": 2, "title": "X-Ray Analysis", "content": "No abnormalities detected in chest X-ray"},
    {"id": 3, "title": "MRI Report", "content": "Minor disc herniation observed in L4-L5 region"}
]


def send_patient_update():
    patient = random.choice(sample_patients)
    producer.send('patient_updates', value=patient)
    print(f"Sent patient update: {patient}")


def send_report_update():
    report = random.choice(sample_reports)
    producer.send('report_updates', value=report)
    print(f"Sent report update: {report}")


def main():
    try:
        while True:
            # Randomly choose to send either a patient update or a report update
            if random.choice([True, False]):
                send_patient_update()
            else:
                send_report_update()

            # Wait for a random time between 1 and 5 seconds
            time.sleep(random.uniform(1, 5))
    except KeyboardInterrupt:
        print("Stopping producer...")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
