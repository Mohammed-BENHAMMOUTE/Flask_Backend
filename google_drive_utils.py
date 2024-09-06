import os
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import logging
from langchain.schema import Document
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import necessary components from your main application
from app import embedding, vectorstore, text_splitter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update scope to allow managing Drive files
SCOPES = ['https://www.googleapis.com/auth/drive']

# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = os.getenv("CLIENT_SECRETS_FILE")
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

logger.info(f"Using folder ID: {FOLDER_ID}")


def get_drive_service():
    logger.info(f"Service account file path: {SERVICE_ACCOUNT_FILE}")
    logger.info(f"File exists: {os.path.exists(SERVICE_ACCOUNT_FILE)}")
    logger.info(f"File is readable: {os.access(SERVICE_ACCOUNT_FILE, os.R_OK)}")

    try:
        with open(SERVICE_ACCOUNT_FILE, 'r') as f:
            service_account_info = json.load(f)
            logger.info(f"Service account type: {service_account_info.get('type')}")
            logger.info(f"Service account client_email: {service_account_info.get('client_email')}")
    except Exception as e:
        logger.error(f"Error reading service account file: {str(e)}")

    creds = None
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        logger.info("Credentials created successfully")
    except Exception as e:
        logger.error(f"Error loading service account file: {str(e)}")
        raise

    try:
        service = build('drive', 'v3', credentials=creds)
        logger.info("Drive service built successfully")
        return service
    except Exception as e:
        logger.error(f"Error building drive service: {str(e)}")
        raise


def list_unprocessed_pdf_files(drive_service):
    # First, let's check all files in the folder
    all_files_query = f"'{FOLDER_ID}' in parents"
    all_files_results = drive_service.files().list(
        q=all_files_query,
        pageSize=1000,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    all_files = all_files_results.get('files', [])
    logger.info(f"Total files found in folder: {len(all_files)}")
    for file in all_files:
        logger.info(f"File: {file['name']}, Type: {file['mimeType']}")

    # Now, let's check for PDF files
    pdf_query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf'"
    pdf_results = drive_service.files().list(
        q=pdf_query,
        pageSize=1000,
        fields="nextPageToken, files(id, name)"
    ).execute()
    pdf_files = pdf_results.get('files', [])
    logger.info(f"PDF files found in folder: {len(pdf_files)}")

    # Finally, let's check for unprocessed PDF files
    unprocessed_query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and not name contains 'processed_'"
    unprocessed_results = drive_service.files().list(
        q=unprocessed_query,
        pageSize=1000,
        fields="nextPageToken, files(id, name)"
    ).execute()
    unprocessed_files = unprocessed_results.get('files', [])
    logger.info(f"Unprocessed PDF files found in folder: {len(unprocessed_files)}")

    return unprocessed_files



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
    logger.info(f"Starting to process PDFs from Drive at {datetime.now()}")
    try:
        drive_service = get_drive_service()
        files = list_unprocessed_pdf_files(drive_service)
        logger.info(f"Found {len(files)} unprocessed PDF files")

        for file in files:
            logger.info(f"Processing file: {file['name']}")
            try:
                pdf_content = download_file(drive_service, file['id'])
                text = extract_text_from_pdf(pdf_content)

                doc = Document(
                    page_content=text,
                    metadata={"source": f"google_drive_{file['id']}", "id": file['id'], "type": "pdf"}
                )
                splits = text_splitter.split_documents([doc])

                vectorstore.add_documents(splits)
                logger.info(f"Added {len(splits)} document chunks to the vector store for file: {file['name']}")

                new_name = f"processed_{file['name']}"
                rename_file(drive_service, file['id'], new_name)
                logger.info(f"Renamed file to: {new_name}")
            except Exception as e:
                logger.error(f"Error processing file {file['name']}: {str(e)}")

        logger.info(f"Processed {len(files)} files from Google Drive.")
    except Exception as e:
        logger.error(f"Error in process_pdfs_from_drive: {str(e)}")

if __name__ == "__main__":
    process_pdfs_from_drive()