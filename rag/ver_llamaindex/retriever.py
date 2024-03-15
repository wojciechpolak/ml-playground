"""
# ml-playground/rag/ver_llamaindex/retriever
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

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env


logger = configure_logging('retriever')
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


def load_data(chroma_collection) -> VectorStoreIndex:
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, ssl=False)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )


if __name__ == '__main__':
    index = load_data(chroma_collection)

    if len(sys.argv) < 2:
        query = input('Query: ')
    else:
        query = sys.argv[1]
    logger.debug('QUERY: %s', query)

    use_query_engine = env.QE
    if use_query_engine:
        Settings.llm = None
        query_engine = index.as_query_engine()
        res = query_engine.query(query)
        pprint.pprint(res, indent=2, width=120)
    else:
        retrieve_engine = index.as_retriever(similarity_top_k=env.TOP_K)
        res = retrieve_engine.retrieve(query)
        for r in res:
            pprint.pprint(r.to_dict(), indent=2, width=120)
