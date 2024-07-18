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
        collection_name="medical_documents",
        use_jsonb=True
    )
    logger.info("Successfully connected to the vector store.")
except Exception as e:
    logger.error(f"Failed to connect to the vector store: {str(e)}")
    raise

# Load PDFs and add to vectorstore at startup
def load_pdfs_to_vectorstore():
    pdf_docs = load_pdf_docs()
    if pdf_docs:
        splits = text_splitter.split_documents(pdf_docs)
        vectorstore.add_documents(splits)
        logger.info(f"Loaded {len(splits)} document chunks from PDFs into the vector store.")
    else:
        logger.warning("No PDF documents found in the specified folder.")

# Call this function at startup
load_pdfs_to_vectorstore()

# Create a retriever
retriever = vectorstore.as_retriever()

# Create a chat model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Create a conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# Define context
context ="""
 Vous êtes un assistant médical avancé conçu pour aider les médecins et professionnels de santé au Maroc. Votre base de connaissances couvre la médecine générale et spécialisée, avec une expertise particulière sur les pratiques et médicaments marocains.

Objectif : Fournir des informations médicales précises, à jour et pertinentes pour aider les professionnels de santé dans leur pratique quotidienne.

Directives :
1. Répondez en français, sauf demande contraire.
2. Adaptez la longueur et le détail de vos réponses selon la complexité de la question.
3. Basez vos réponses sur les dernières preuves médicales et directives cliniques.
4. Intégrez des informations spécifiques au contexte médical marocain quand c'est pertinent.
5. Fournissez des explications claires sur les raisonnements diagnostiques et les recommandations de traitement.
6. Incluez des conseils sur la prévention et la gestion des maladies si approprié.
7. Vous pouvez suggérer des diagnostics potentiels et des traitements, y compris des médicaments marocains spécifiques.

Ton : Professionnel, précis et scientifique, tout en restant accessible et collaboratif.

Rappels :
- Bien que vos réponses soient basées sur des informations médicales solides, rappelez aux utilisateurs l'importance du jugement clinique et de l'évaluation individuelle de chaque patient.
- Soulignez l'importance de la confidentialité des informations médicales.

Votre rôle est d'être une ressource fiable et un support pour les professionnels de santé, en les aidant à prendre des décisions éclairées pour leurs patients.
"""
# Initialize PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(CONNECTION_STRING)

@app.route('/chatbot/ask', methods=['POST'])
def chat():
    try:
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

@app.route('/upsert-client-embedding', methods=['POST'])
def upsert_client_embedding():
    try:
        client = request.json
        if not client:
            return jsonify({"error": "No client data provided"}), 400
        
        client_id = str(client['id'])
        client_text = f"{client['name']} {client['details']}"
        
        # Create a Document object
        doc = Document(
            page_content=client_text,
            metadata={"source": f"client_{client_id}", "id": client_id}
        )
        
        # Split the document (even though it's just one, to maintain consistency with your existing code)
        splits = text_splitter.split_documents([doc])
        
        # Add the document to the vectorstore
        vectorstore.add_documents(splits)
        
        logger.info(f"Upserted embedding for client ID: {client_id}")
        return jsonify({"status": "success", "message": "Client embedding upserted"}), 200
    except Exception as e:
        logger.error(f"Error in upsert_client_embedding endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

# Add this new route to your existing Flask app


# Add this new route to your existing Flask app
@app.route('/load_pdfs', methods=['POST'])
def load_pdfs():
    try:
        load_pdfs_to_vectorstore()
        return jsonify({"message": "PDFs processed and added to the vector store"}), 200
    except Exception as e:
        logger.error(f"Error in load_pdfs endpoint: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")