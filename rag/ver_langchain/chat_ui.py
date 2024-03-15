"""
# ml-playground/rag/ver_langchain/chat_ui
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

import pprint
from os.path import basename

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_debug, set_verbose
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from streamlit.components.v1 import html

from rag.utils.audio import load_audio
from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env


logger = configure_logging('main')
log_env(logger)

set_debug(env.DEBUG > 1)
set_verbose(env.VERBOSE)

if 'initiated' not in st.session_state:
    st.session_state.initiated = True

st.set_page_config(page_title=env.APP_TITLE)
st.title(env.APP_TITLE)


def change_collection():
    st.cache_resource.clear()
    chat_history.clear()


top_k = env.TOP_K
ollama_temperature = env.OLLAMA_TEMPERATURE
collections = env.COLLECTION_NAME.split(',')

if len(collections) > 1:
    collection_name = st.selectbox(
        'Select knowledge base',
        collections,
        on_change=change_collection)
else:
    collection_name = env.COLLECTION_NAME

with st.expander('Advanced', expanded=False):
    ollama_temperature = st.slider(
        'Temperature', 0.0, 1.0, env.OLLAMA_TEMPERATURE, step=0.01
    )
    top_k = st.slider(
        'Similarity Top K', 1, 10, env.TOP_K, step=1
    )

# create client and a new collection
chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(collection_name)

if env.OLLAMA_EMBEDDING:
    embed_model = OllamaEmbeddings(
        model_name=env.OLLAMA_EMBEDDING,
        base_url=env.OLLAMA_HOST
    )
else:
    embed_model = HuggingFaceEmbeddings(
        model_name=env.HF_EMBEDDING,
    )

chat_history = StreamlitChatMessageHistory(key='chat_messages')

# Initialize the chat message history
if len(chat_history.messages) == 0:
    chat_history.add_ai_message('Ask me a question!')


@st.cache_resource(show_spinner=False)
def load_data() -> Chroma:
    logger.info('Loading data from collection: %s', chroma_collection.name)
    with st.spinner(text='Loading the index...'):
        return Chroma(
            client=chroma_client,
            collection_name=env.COLLECTION_NAME,
            embedding_function=embed_model,
        )


index = load_data()
retriever = index.as_retriever(
    search_type='similarity',
    search_kwargs={'k': top_k}
)

llm = ChatOllama(
    model=env.OLLAMA_MODEL,
    request_timeout=120.0,
    temperature=ollama_temperature,
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

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

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

# Create the chain
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Wrap with message history
chat_history = StreamlitChatMessageHistory(key='chat_messages')

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

for message in chat_history.messages:
    if message.type == 'human':
        role = 'user'
    elif message.type == 'ai':
        role = 'assistant'
    else:
        role = 'system'
    with st.chat_message(role):
        st.write(message.content)

if 'audio_bytes_hash' not in st.session_state:
    st.session_state.audio_bytes_hash = None

prompt_audio = ''
audio_bytes = audio_recorder(text='', icon_size='2x', energy_threshold=0)
if audio_bytes and hash(audio_bytes) != st.session_state.audio_bytes_hash:
    st.session_state.audio_bytes_hash = hash(audio_bytes)
    with st.spinner('Processing...'):
        audio_data = load_audio(audio_bytes)
        audio_model = WhisperModel('medium')
        segments, _ = audio_model.transcribe(
            audio_data, task='translate', language=env.WHISPER_LANG)
        segments = list(segments)
        logger.debug('Whisper segments: ', segments)
        texts = [obj.text for obj in segments]
        prompt_audio = ' '.join(texts).strip()
        logger.info('Transcribe: %s', prompt_audio)

if prompt_audio:
    prompt = prompt_audio
    st.chat_input('Your question')
elif prompt := st.chat_input('Your question'):
    pass

if prompt:
    with st.chat_message('user'):
        st.write(prompt)


if 'speakOutput' not in st.session_state:
    st.session_state.speakOutput = False


def generate_speech_js():
    text = str(st.session_state.speakOutput)
    text = text.replace('\n', ' ')
    text = text.replace('"', "'")
    html(f"""<script>
        let utterance = new SpeechSynthesisUtterance("{text}");
        utterance.voice = speechSynthesis.getVoices().filter(i => i.default)[0];
        utterance.rate = 0.9;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
        </script>""")
    st.session_state.speakOutput = False


def speak_output(data):
    st.session_state.speakOutput = data

# If last message is not from assistant, generate a new response
if prompt:
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            # del st.session_state.messages[-1]  # weird dup bug
            config = {
                'configurable': {'session_id': 'abc123'}
            }
            response = conversational_rag_chain.invoke(
                {'input': prompt},
                config,
            )

            if env.DEBUG:
                logger.debug('PROMPTS:')
                pprint.pprint(conversational_rag_chain.get_prompts(config), indent=2, width=150)

            if env.PDF_INCLUDE_SOURCE and response['context'][0].metadata:
                meta_info = []
                metadata = response['context'][0].metadata
                logger.debug(f"Metadata:\t {metadata}")
                if 'source' in metadata:
                    file_name = basename(metadata.get('source', ''))
                    is_pdf = '.pdf' in file_name.lower()
                    if is_pdf:
                        meta_info.append(f'* [{file_name}]({env.PDF_SOURCE_LINK_PREFIX}{file_name.replace(" ", "%20")})')
                meta_info = list(set(meta_info))
                st.write(response['answer'] + '\n\n' + '\n'.join(meta_info))
            else:
                st.write(response['answer'])

            if env.DEBUG:
                logger.debug('---CHAT HISTORY---')
                pprint.pprint(chat_history.messages, indent=2, width=150)
                logger.debug('---END HISTORY---')

    # Button to trigger speechSynthesis
    st.button(':speaking_head_in_silhouette: Read Aloud',
              on_click=speak_output,
              kwargs=dict(data=response['answer']))

if st.session_state.speakOutput:
    generate_speech_js()
