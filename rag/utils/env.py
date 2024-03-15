"""
# ml-playground/rag/utils/env
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

import warnings
warnings.simplefilter('error')

import os
from dotenv import load_dotenv

load_dotenv()


class Env:
    APP_TITLE = os.getenv('APP_TITLE', 'My AI Assistant')
    CHROMA_HOST = os.getenv('CHROMA_HOST')
    CHROMA_PATH = os.getenv('CHROMA_PATH', './run/chroma_db')
    CHROMA_PORT = os.getenv('CHROMA_PORT')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'default')
    COLLECTION_RESET = os.getenv('COLLECTION_RESET', False)
    DEBUG = bool(os.getenv('DEBUG', False))
    HF_EMBEDDING = os.getenv('HF_EMBEDDING', 'intfloat/multilingual-e5-small')
    INDEXER_LOADER = os.getenv('INDEXER_LOADER', 'dir')
    INDEXER_SPLIT_DOCS = os.getenv('INDEXER_SPLIT_DOCS', False)
    OLLAMA_EMBEDDING = os.getenv('OLLAMA_EMBEDDING')
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    OLLAMA_TEMPERATURE = float(os.getenv('OLLAMA_TEMPERATURE', 0.75))
    PDF_INCLUDE_SOURCE = os.getenv('PDF_INCLUDE_SOURCE', False)
    PDF_SOURCE_LINK_PREFIX = os.getenv('PDF_SOURCE_LINK_PREFIX', '')
    QE = os.getenv('QE', False)  # Use Query Engine
    TOP_K = int(os.getenv('TOP_K', 5))
    WHISPER_LANG = os.getenv('WHISPER_LANG', None)
    VERBOSE = bool(os.getenv('VERBOSE', False))
    WL_BEARER_TOKEN = os.getenv('WL_BEARER_TOKEN')
    WL_COOKIE_TOKEN = os.getenv('WL_COOKIE_TOKEN')
    WL_DEPTH_LIMIT = int(os.getenv('WL_DEPTH_LIMIT', 0))

env = Env()
