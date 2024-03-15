"""
# ml-playground/rag/utils/logging_config
#
# Copyright (C) 2024 Wojciech Polak
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import sys
import logging
from .env import env


def configure_logging(logger_name: str = 'default') -> logging.Logger:
    logging.basicConfig(
        stream=sys.stdout,
        # level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if env.DEBUG else logging.INFO)
    return logger


def log_env(logger: logging.Logger):
    logger.info('OLLAMA_HOST: %s', env.OLLAMA_HOST)
    logger.info('OLLAMA_MODEL: %s', env.OLLAMA_MODEL)
    logger.info('OLLAMA_TEMPERATURE: %s', env.OLLAMA_TEMPERATURE)
    logger.info('EMBEDDING: %s',
                env.OLLAMA_EMBEDDING if env.OLLAMA_EMBEDDING else env.HF_EMBEDDING)
    logger.info('COLLECTION_NAME: %s', env.COLLECTION_NAME)
    logger.info('TOP_K: %s', env.TOP_K)
