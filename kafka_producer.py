
import json
from kafka import KafkaProducer
from dotenv import load_dotenv
import os

load_dotenv()


class TestKafkaProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_patient_update(self, patient_data):
        self.producer.send('patient_updates', patient_data)
        self.producer.flush()

    def send_report_update(self, report_data):
        self.producer.send('report_updates', report_data)
        self.producer.flush()

    def close(self):
        self.producer.close()