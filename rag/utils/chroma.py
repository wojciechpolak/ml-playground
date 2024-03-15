"""
# ml-playground/rag/utils/chroma
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

import chromadb
from chromadb.config import Settings as ChromaSettings
from .logging_config import configure_logging
from .env import env

logger = configure_logging('indexer')


def get_chroma_client() -> chromadb.ClientAPI:
    if env.CHROMA_HOST:
        chroma_client = chromadb.HttpClient(
            host=env.CHROMA_HOST, port=env.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False))
    else:
        chroma_client = chromadb.PersistentClient(
            path=env.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False))

    if env.COLLECTION_RESET:
        logger.info('Resetting collection...')
        try:
            chroma_client.delete_collection(env.COLLECTION_NAME)
            logger.debug('list_collections: %s', chroma_client.list_collections())
        except Exception as e:
            logger.error('Error: %s', e)

    return chroma_client
