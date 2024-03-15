"""
# ml-playground/rag/ver_langchain/chat_text
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

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_debug, set_verbose
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env


logger = configure_logging('main')
log_env(logger)

set_debug(env.DEBUG > 1)
set_verbose(env.VERBOSE)

chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(env.COLLECTION_NAME)

def load_data() -> Chroma:
    if env.OLLAMA_EMBEDDING:
        embed_model = OllamaEmbeddings(
            model=env.OLLAMA_EMBEDDING,
            base_url=env.OLLAMA_HOST
        )
    else:
        embed_model = HuggingFaceEmbeddings(
            model_name=env.HF_EMBEDDING,
        )

    logger.info('Loading data from collection: %s', chroma_collection.name)
    return Chroma(
        client=chroma_client,
        collection_name=env.COLLECTION_NAME,
        embedding_function=embed_model,
    )


index = load_data()
retriever = index.as_retriever(
    search_type='similarity',
    search_kwargs={'k': env.TOP_K}
)

llm = ChatOllama(
    model=env.OLLAMA_MODEL,
    request_timeout=120.0,
    temperature=env.OLLAMA_TEMPERATURE,
    base_url=env.OLLAMA_HOST)


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

CONTEXT:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', qa_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {}
chat_history = ChatMessageHistory(key='chat_messages')

# Initialize the chat message history
if len(chat_history.messages) == 0:
    chat_history.add_ai_message('Ask me a question!')


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = chat_history
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)


def repl(initial_query: str | None):
    print('Enter your queries (press Ctrl-D to exit):')
    while True:
        try:
            query = initial_query or input('PROMPT >>> ')
            initial_query = None
            if not query:
                continue
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL.")
            break
        try:
            res = conversational_rag_chain.invoke(
                {'input': query},
                config={
                    'configurable': {'session_id': 'abc123'}
                },
            )
            if env.DEBUG:
                logger.debug('---CHAT HISTORY---')
                for msg in chat_history:
                    pprint.pprint(msg, indent=2, width=120)
                logger.debug('---END HISTORY---')
            else:
                print(res['answer'])
        except Exception as exc:
            logger.error('Exception: %s', exc)


if __name__ == '__main__':
    query = None
    if len(sys.argv) == 2:
        query = sys.argv[1]
    repl(query)
