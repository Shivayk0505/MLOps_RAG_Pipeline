import mlflow
from datetime import datetime

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("rag_system_experiments")

def log_rag_run(pdf_name, chunks_count, embedding_model, llm_model, latency_ms):
    with mlflow.start_run(run_name=f"RAG_{pdf_name}_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param("pdf_name", pdf_name)
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("llm_model", llm_model)
        mlflow.log_metric("num_chunks", chunks_count)
        mlflow.log_metric("processing_latency_ms", latency_ms)
