import logging
import os

# Create a 'logs' directory inside the server folder if it doesn't exist
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SERVER_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define the exact path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, "server.log")

# Setup the global logging configuration
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG if you want to see detailed chunk previews
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # 1. This handler saves all logs inside the server/logs/server.log file
        logging.FileHandler(LOG_FILE_PATH),
        
        # 2. This handler keeps showing the logs live on your terminal console
        logging.StreamHandler()
    ]
)

# Export the configured logger
logger = logging.getLogger("DocuMind")