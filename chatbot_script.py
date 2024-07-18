import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from dotenv import load_dotenv
import logging
from functools import lru_cache
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# PostgreSQL connection string
CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

# Create embeddings
embedding = OpenAIEmbeddings()

# Load PDFs
pdf_folder_path = os.getenv("PDF_FOLDER_PATH")

@lru_cache(maxsize=None)
def load_pdf_docs():
    pdf_docs = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
            pdf_docs.extend(loader.load())
    return pdf_docs

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize vectorstore
try:
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embedding,
        collection_name="medical_documents"
    )
    logger.info("Successfully connected to the vector store.")
except Exception as e:
    logger.error(f"Failed to connect to the vector store: {str(e)}")
    raise

# Create a retriever
retriever = vectorstore.as_retriever()

# Create a chat model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Create a conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# Define context
context = """
Tâche : Répondre aux questions des médecins concernant les connaissances médicales et les informations sur les patients. Identifier les symptômes, proposer des diagnostics potentiels et prescrire des médicaments marocains. Préciser que les suggestions ne sont pas 100% précises et doivent être validées par un professionnel de santé.

Persona : Vous êtes un expert en santé familiarisé avec la médication marocaine et les pratiques médicales internationales. Vous possédez une vaste connaissance en médecine générale et spécialisée.

Langue : Répondez en français, sauf si on vous demande explicitement d'utiliser une autre langue.

Format :
- Fournissez des réponses concises (4 lignes maximum) en points clairs, sauf si plus de détails sont demandés.
- Élaborez jusqu'à 1000 lignes si des informations détaillées sont requises.
- Structurez vos réponses de manière logique et facile à lire.

Ton :
- Professionnel mais bienveillant.
- Langage clair, précis et élégant.
- Empathique envers les préoccupations des patients et des médecins.

Contenu :
- Basez vos réponses sur les dernières preuves médicales et directives cliniques.
- Intégrez, le cas échéant, des informations spécifiques au contexte médical marocain.
- Fournissez des explications claires sur les raisonnements diagnostiques et les recommandations de traitement.
- Incluez des conseils sur la prévention et la gestion des maladies lorsque c'est pertinent.

Rappel important :
- Spécifiez toujours que les suggestions fournies ne sont pas définitives et nécessitent la supervision d'un professionnel de santé.
- Encouragez la consultation d'un médecin pour un diagnostic et un traitement précis.
- Soulignez l'importance de la confidentialité des informations médicales.

Limites :
- Ne posez pas de diagnostic définitif.
- N'encouragez pas l'automédication.
- Ne fournissez pas d'informations médicales controversées ou non prouvées.

Objectif : Être un assistant médical fiable, informatif et éthique, aidant les médecins dans leur pratique quotidienne tout en promouvant des soins de santé sûrs et de haute qualité.
"""

# Initialize PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(CONNECTION_STRING)

def create_patients_table_if_not_exists():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    age INTEGER,
                    medical_history TEXT,
                    last_updated TIMESTAMP,
                    last_embedded TIMESTAMP
                )
            """)
        conn.commit()
    logger.info("Patients table created or already exists.")

# Function to embed new patient data
def embed_new_patient(patient_data):
    doc = Document(
        page_content=f"Patient ID: {patient_data['id']}\n"
                     f"Name: {patient_data['name']}\n"
                     f"Age: {patient_data['age']}\n"
                     f"Medical History: {patient_data['medical_history']}",
        metadata={"patient_id": patient_data['id']}
    )
    
    vectorstore.add_documents([doc])
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE patients SET last_embedded = %s WHERE id = %s",
                (datetime.now(), patient_data['id'])
            )
        conn.commit()

# Function to update existing patient embedding
def update_patient_embedding(patient_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT * FROM patients WHERE id = %s", (patient_id,))
            patient_data = cur.fetchone()
    
    if patient_data:
        vectorstore.delete(filter={"patient_id": patient_id})
        embed_new_patient(patient_data)
    else:
        logger.warning(f"Patient with ID {patient_id} not found.")

# Function to check and update patient embeddings
def check_and_update_embeddings():
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT * FROM patients 
                    WHERE last_embedded IS NULL 
                    OR last_embedded < last_updated
                """)
                patients_to_update = cur.fetchall()
        
        for patient in patients_to_update:
            embed_new_patient(patient)
    except psycopg2.errors.UndefinedTable:
        logger.warning("Patients table does not exist. Creating it now.")
        create_patients_table_if_not_exists()
    except Exception as e:
        logger.error(f"Error in check_and_update_embeddings: {str(e)}")

@app.route('/chatbot/ask', methods=['POST'])
def chat():
    try:
        check_and_update_embeddings()
        
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f'Received message: {message}')
        
        docs = retriever.get_relevant_documents(message)
        
        retrieved_context = "\n".join([doc.page_content for doc in docs])
        
        combined_context = f"{context}\n\nRelevant Information:\n{retrieved_context}\n\nQ: {message}\nA:"
        
        response = qa_chain({"question": combined_context, "chat_history": []})
        
        return jsonify({"response": response['answer']})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/add_patient', methods=['POST'])
def add_patient():
    try:
        patient_data = request.json
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO patients (name, age, medical_history, last_updated)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (patient_data['name'], patient_data['age'], 
                      patient_data['medical_history'], datetime.now()))
                patient_id = cur.fetchone()[0]
            conn.commit()
        
        patient_data['id'] = patient_id
        embed_new_patient(patient_data)
        
        return jsonify({"message": "Patient added and embedded successfully", "id": patient_id}), 200
    except Exception as e:
        logger.error(f"Error in add_patient endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/update_patient/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    try:
        patient_data = request.json
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE patients 
                    SET name = %s, age = %s, medical_history = %s, last_updated = %s
                    WHERE id = %s
                """, (patient_data['name'], patient_data['age'], 
                      patient_data['medical_history'], datetime.now(), patient_id))
            conn.commit()
        
        update_patient_embedding(patient_id)
        
        return jsonify({"message": "Patient updated and re-embedded successfully"}), 200
    except Exception as e:
        logger.error(f"Error in update_patient endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/add_json_data', methods=['POST'])
def add_json_data():
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        documents = [Document(page_content=str(value), metadata={"key": key}) for key, value in data.items()]
        
        splits = text_splitter.split_documents(documents)
        
        vectorstore.add_documents(splits)
        
        return jsonify({"message": "JSON data added successfully"}), 200
    except Exception as e:
        logger.error(f"Error in add_json_data endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/load_pdfs', methods=['POST'])
def load_pdfs():
    try:
        pdf_docs = load_pdf_docs()
        if not pdf_docs:
            logger.warning("No PDF documents found in the specified folder.")
            return jsonify({"message": "No PDF documents found"}), 200
        
        splits = text_splitter.split_documents(pdf_docs)
        vectorstore.add_documents(splits)
        return jsonify({"message": f"Processed and added {len(splits)} document chunks to the vector store"}), 200
    except Exception as e:
        logger.error(f"Error in load_pdfs endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

# Create patients table if it doesn't exist
create_patients_table_if_not_exists()

if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")