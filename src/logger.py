from loguru import logger
import os

# Setup logging folder
os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/rag_system.log",
    rotation="1 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def get_logger():
    return logger
