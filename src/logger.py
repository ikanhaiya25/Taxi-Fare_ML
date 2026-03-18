import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir,"project.log"),
    level=logging.INFO,
    format="[%(asctime)s] %(message)s"
)