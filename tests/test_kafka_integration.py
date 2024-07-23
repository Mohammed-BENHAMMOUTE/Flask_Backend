import unittest
import time
from kafka_producer import TestKafkaProducer
from app import app, vectorstore, embedding


class TestKafkaIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.producer = TestKafkaProducer()

    def tearDown(self):
        self.producer.close()

    def test_patient_update(self):
        patient_data = {
            "id": "test_patient_1",
            "name": "John Doe",
            "details": "35-year-old male with a history of hypertension"
        }
        self.producer.send_patient_update(patient_data)

        # Wait for message processing
        time.sleep(5)

        # Check if the patient data was added to the vectorstore
        results = vectorstore.similarity_search(
            "John Doe hypertension",
            k=1,
            filter={"source": f"profile_{patient_data['id']}"}
        )
        self.assertEqual(len(results), 1)
        self.assertIn(patient_data['name'], results[0].page_content)
        self.assertIn(patient_data['details'], results[0].page_content)

    def test_report_update(self):
        report_data = {
            "id": "test_report_1",
            "title": "Annual Checkup",
            "content": "Patient shows normal vital signs and good overall health."
        }
        self.producer.send_report_update(report_data)

        # Wait for message processing
        time.sleep(5)

        # Check if the report data was added to the vectorstore
        results = vectorstore.similarity_search(
            "Annual Checkup vital signs",
            k=1,
            filter={"source": f"report_{report_data['id']}"}
        )
        self.assertEqual(len(results), 1)
        self.assertIn(report_data['title'], results[0].page_content)
        self.assertIn(report_data['content'], results[0].page_content)

    def test_dlq_processing(self):
        # Simulate a failed message by sending to DLQ directly
        failed_patient_data = {
            "id": "failed_patient_1",
            "name": "Jane Smith",
            "details": "42-year-old female with a history of diabetes"
        }
        self.producer.producer.send('patient_updates_dlq', failed_patient_data)
        self.producer.producer.flush()

        # Wait for DLQ processing
        time.sleep(10)

        # Check if the failed patient data was processed from DLQ and added to vectorstore
        results = vectorstore.similarity_search(
            "Jane Smith diabetes",
            k=1,
            filter={"source": f"profile_{failed_patient_data['id']}"}
        )
        self.assertEqual(len(results), 1)
        self.assertIn(failed_patient_data['name'], results[0].page_content)
        self.assertIn(failed_patient_data['details'], results[0].page_content)

if __name__ == '__main__':
    unittest.main()