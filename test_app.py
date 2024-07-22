import unittest
from unittest.mock import patch, MagicMock
import json
import app


class TestMedicalAssistantApp(unittest.TestCase):
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    @patch('app.qa_chain')
    @patch('app.retriever.get_relevant_documents')
    def test_chatbot_ask(self, mock_get_relevant_documents, mock_qa_chain):
        # Mock the retriever and qa_chain
        mock_get_relevant_documents.return_value = [
            MagicMock(page_content="Test content", metadata={"type": "profile"})
        ]
        mock_qa_chain.return_value = {"answer": "Test response"}

        # Test the chatbot endpoint
        response = self.app.post('/chatbot/ask',
                                 data=json.dumps({'message': 'Test question'}),
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertEqual(data['response'], 'Test response')

    @patch('app.load_pdfs_to_vectorstore')
    def test_load_pdfs(self, mock_load_pdfs):
        # Test the load_pdfs endpoint
        response = self.app.post('/load_pdfs')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'PDFs processed and added to the vector store')

    @patch('app.KafkaConsumer')
    @patch('app.KafkaProducer')
    @patch('app.upsert_patient_embedding')
    def test_process_patient_messages(self, mock_upsert, mock_producer, mock_consumer):
        # Mock Kafka consumer
        mock_consumer.return_value = MagicMock()
        mock_consumer.return_value.__iter__.return_value = [
            MagicMock(value=json.dumps({"id": 1, "name": "Test Patient", "details": "Test details"}))
        ]

        # Run the patient message processing
        app.process_patient_messages()

        # Check if upsert_patient_embedding was called
        mock_upsert.assert_called_once()

    @patch('app.KafkaConsumer')
    @patch('app.KafkaProducer')
    @patch('app.upsert_report_embedding')
    def test_process_report_messages(self, mock_upsert, mock_producer, mock_consumer):
        # Mock Kafka consumer
        mock_consumer.return_value = MagicMock()
        mock_consumer.return_value.__iter__.return_value = [
            MagicMock(value=json.dumps({"id": 1, "title": "Test Report", "content": "Test content"}))
        ]

        # Run the report message processing
        app.process_report_messages()

        # Check if upsert_report_embedding was called
        mock_upsert.assert_called_once()

    @patch('app.vectorstore')
    def test_upsert_patient_embedding(self, mock_vectorstore):
        # Test upsert_patient_embedding function
        profile = {"id": 1, "name": "Test Patient", "details": "Test details"}
        app.upsert_patient_embedding(profile)

        # Check if vectorstore.add_documents was called
        mock_vectorstore.add_documents.assert_called_once()

    @patch('app.vectorstore')
    def test_upsert_report_embedding(self, mock_vectorstore):
        # Test upsert_report_embedding function
        report = {"id": 1, "title": "Test Report", "content": "Test content"}
        app.upsert_report_embedding(report)

        # Check if vectorstore.add_documents was called
        mock_vectorstore.add_documents.assert_called_once()


if __name__ == '__main__':
    unittest.main()