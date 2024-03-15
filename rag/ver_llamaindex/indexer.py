"""
# ml-playground/rag/ver_llamaindex/indexer
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
from urllib.parse import urlparse

from llama_index.core import VectorStoreIndex, Settings, \
    SimpleDirectoryReader, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env
from rag.utils.scrape import get_links


logger = configure_logging('indexer')
log_env(logger)

chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(env.COLLECTION_NAME)

if env.OLLAMA_EMBEDDING:
    Settings.embed_model = OllamaEmbedding(
        model=env.OLLAMA_EMBEDDING,
        base_url=env.OLLAMA_HOST
    )
else:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=env.HF_EMBEDDING,
    )


def index_local_data(input_dir: str) -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        exclude=[
            '**/*.canvas',
            '**/*.json',
            '**/*.png',
            '**/*.jpg',
            '**/*.hif',
            '**/*.webp',
        ],
        filename_as_id=True,
    )
    docs = reader.load_data(show_progress=True)

    logger.debug('Number of docs: %s', len(docs))
    for doc in docs:
        logger.info('DOC >>> %s : %s',
                    doc.text[:60].replace('\n', ' '),
                    doc.metadata.get('file_name'))
    return docs


def index_web_data(urls: list[str]) -> list[Document]:
    reader = SimpleWebPageReader(
        html_to_text=True,
        metadata_fn=lambda src: {'url_src': src})
    docs = reader.load_data(urls)
    for doc in docs:
        logger.info('DOC >>> %s : %s',
                    doc.text[:60].replace('\n', ' '),
                    doc.metadata.get('url_src'))
    return docs


def store_documents(docs: list[Document]):
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
        collection_name=env.COLLECTION_NAME,
        ssl=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True
    )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} input-src')
        sys.exit(1)

    input_src = sys.argv[1]

    if input_src.startswith('http'):
        start_url = input_src
        domain = urlparse(start_url).netloc
        urls = set(get_links(start_url, domain, env.WL_DEPTH_LIMIT))
        urls.add(start_url)
        urls = list(urls)

        logger.info('Links from the same domain:')
        for url in urls:
            logger.info(url)

        docs = index_web_data(urls)
    else:
        docs = index_local_data(input_src)
    store_documents(docs)
