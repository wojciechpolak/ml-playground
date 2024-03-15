"""
# ml-playground/rag/ver_langchain/indexer
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

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, ObsidianLoader, WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env
from rag.utils.scrape import get_links


logger = configure_logging('indexer')
log_env(logger)

chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(env.COLLECTION_NAME)

# Initialize embeddings model
if env.OLLAMA_EMBEDDING:
    embed_model = OllamaEmbeddings(
        model=env.OLLAMA_EMBEDDING,
        show_progress=True)
else:
    embed_model = HuggingFaceEmbeddings(
        model_name=env.HF_EMBEDDING,
        # model_kwargs={'local_files_only': True},
        show_progress=True)


def index_local_data(input_dir: str):
    if env.INDEXER_LOADER == 'obsidian':
        loader = ObsidianLoader(input_dir)
    elif env.INDEXER_LOADER == 'pdf':
        loader = PyPDFDirectoryLoader(input_dir)
    else:
        loader = DirectoryLoader(
            path=input_dir,
            recursive=True,
            # glob='**/*.md',
            exclude=[
                '**/*.canvas',
                '**/*.json',
                '**/*.png',
                '**/*.jpg',
                '**/*.hif',
                '**/*.webp',
            ],
            show_progress=True,
        )

    #
    # https://python.langchain.com/docs/how_to/#text-splitters
    #

    if env.INDEXER_SPLIT_DOCS:
        documents = loader.load_and_split()
    else:
        documents = loader.load()

    logger.info('Number of docs: %s', len(documents))
    for doc in documents:
        logger.info('DOC >>> %s : %s',
                    doc.page_content[:60].replace('\n', ' '),
                    doc.metadata.get('source'))
    return documents


def index_web_data(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for doc in docs:
        logger.info('DOC >>> %s : %s',
                    doc.metadata.get('title', '')[:60].replace('\n', ' '),
                    doc.metadata.get('source'))
    return docs


def store_documents(documents):
    vector_store = Chroma.from_documents(
        client=chroma_client,
        documents=documents,
        embedding=embed_model,
        collection_name=env.COLLECTION_NAME)
    return vector_store


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} input-src')
        sys.exit(1)

    input_src = sys.argv[1]
    logger.debug('Input src: %s', input_src)

    if input_src.startswith('http'):
        start_url = input_src
        domain = urlparse(start_url).netloc
        urls = set(get_links(start_url, domain, env.WL_DEPTH_LIMIT))
        urls.add(start_url)
        urls = list(urls)

        logger.info('Links from the same domain:')
        for url in urls:
            logger.info('Link: %s', url)

        docs = index_web_data(urls)
    else:
        docs = index_local_data(input_src)
    store_documents(docs)
