"""
# ml-playground/rag/ver_langchain/retriever
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
import pprint

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from rag.utils.chroma import get_chroma_client
from rag.utils.logging_config import configure_logging, log_env
from rag.utils.env import env

logger = configure_logging('retriever')
log_env(logger)

chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(env.COLLECTION_NAME)


def load_data() -> Chroma:
    if env.OLLAMA_EMBEDDING:
        embed_model = OllamaEmbeddings(
            model=env.OLLAMA_EMBEDDING,
            show_progress=True)
    else:
        embed_model = HuggingFaceEmbeddings(
            model_name=env.HF_EMBEDDING,
            # model_kwargs={'local_files_only': True},
            show_progress=True)

    vector_store = Chroma(
        client=chroma_client,
        collection_name=env.COLLECTION_NAME,
        embedding_function=embed_model,
    )
    return vector_store


if __name__ == '__main__':
    index = load_data()

    if len(sys.argv) < 2:
        query = input('Query: ')
    else:
        query = sys.argv[1]
    logger.debug('QUERY: %s', query)

    retrieve_engine = index.as_retriever(
        search_type='similarity',
        search_kwargs={'k': env.TOP_K})
    res = retrieve_engine.invoke(query)
    for r in res:
        pprint.pprint(r.model_dump(), indent=2, width=120)
