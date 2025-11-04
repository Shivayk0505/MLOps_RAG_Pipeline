# RAG MLOps MVP

A minimal **Retrieval-Augmented Generation (RAG)** pipeline with MLOps integration.

## Features
- Vector storage using ChromaDB  
- Text embedding with Sentence Transformers  
- Generation via open-source Mistral model  
- Logging and tracking with MLflow  
- Deployable via FastAPI + Docker  

## Setup
```bash
git clone <your-repo>
cd Mlops_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
