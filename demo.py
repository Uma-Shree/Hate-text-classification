from hate.logger import logging
from hate.exception import CustomException


logging.info("Welcome to project")

try:
    a = 7/ 0
except Exception as e:
    raise CustomException(e, sys) from e
