import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

# Import necessary components from your main application
from app import embedding, vectorstore, text_splitter, logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fetch the PostgreSQL connection string from the environment
connection_string = os.getenv('PGVECTOR_CONNECTION_STRING')

if not connection_string:
    logger.error("PGVECTOR_CONNECTION_STRING is not set in the environment variables")
    raise ValueError("PGVECTOR_CONNECTION_STRING is not set")

# Initialize PGVector using the connection string
try:
    vectorstore = PGVector(connection_string=connection_string)
    logger.info("Connected to PGVector vector store successfully.")
except Exception as e:
    logger.error(f"Failed to connect to the vector store: {str(e)}")
    raise

# Define scope to allow managing all Drive files
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = r'C:\Users\ok\Downloads\chatbot-esj-3246e2b7aea5.json'  # Path to the service account JSON key file

def get_drive_service():
    # Authenticate using the service account key
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_unprocessed_pdf_files(drive_service):
    query = "mimeType='application/pdf' and not name contains 'processed_'"
    results = drive_service.files().list(
        q=query,
        fields="nextPageToken, files(id, name)"
    ).execute()
    files = results.get('files', [])
    logger.info(f"Found {len(files)} unprocessed PDF files in Drive")
    return files

def download_file(drive_service, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()
    return file.getvalue()

def rename_file(drive_service, file_id, new_name):
    file = drive_service.files().update(
        fileId=file_id,
        body={'name': new_name}
    ).execute()
    return file

def extract_text_from_pdf(pdf_content):
    reader = PdfReader(BytesIO(pdf_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_pdfs_from_drive():
    logger.info("Starting to process PDFs from Drive")
    drive_service = get_drive_service()
    logger.info("Drive service created")
    files = list_unprocessed_pdf_files(drive_service)
    logger.info(f"Found {len(files)} unprocessed PDF files")

    for file in files:
        logger.info(f"Processing file: {file['name']}")

        pdf_content = download_file(drive_service, file['id'])
        text = extract_text_from_pdf(pdf_content)

        doc = Document(
            page_content=text,
            metadata={"source": f"google_drive_{file['id']}", "id": file['id'], "type": "pdf"}
        )
        splits = text_splitter.split_documents([doc])

        try:
            vectorstore.add_documents(splits)
            logger.info(f"Added {len(splits)} document chunks to the vector store for file: {file['name']}")

            new_name = f"processed_{file['name']}"
            rename_file(drive_service, file['id'], new_name)
            logger.info(f"Renamed file to: {new_name}")
        except Exception as e:
            logger.error(f"Error processing file {file['name']}: {str(e)}")

    logger.info(f"Processed {len(files)} files from Google Drive.")

if __name__ == "__main__":
    process_pdfs_from_drive()
