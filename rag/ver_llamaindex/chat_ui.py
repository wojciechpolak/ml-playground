"""
# ml-playground/rag/ver_llamaindex/chat_ui
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

from os.path import basename

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from streamlit.components.v1 import html

from rag.utils.audio import load_audio
from rag.utils.chroma import get_chroma_client
from rag.utils.env import env
from rag.utils.logging_config import configure_logging, log_env

logger = configure_logging('main')
log_env(logger)

if 'initiated' not in st.session_state:
    st.session_state.initiated = True

st.set_page_config(page_title=env.APP_TITLE)
st.title(env.APP_TITLE)


def change_collection():
    st.cache_resource.clear()
    del st.session_state.messages


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
        'Temperature', 0.0, 1.0, env.OLLAMA_TEMPERATURE, step=0.01)
    top_k = st.slider(
        'Similarity Top K', 1, 10, env.TOP_K, step=1
    )

# create client and a new collection
chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(collection_name)

if env.OLLAMA_EMBEDDING:
    Settings.embed_model = OllamaEmbedding(
        model=env.OLLAMA_EMBEDDING,
        base_url=env.OLLAMA_HOST
    )
else:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=env.HF_EMBEDDING,
    )

Settings.llm = Ollama(model=env.OLLAMA_MODEL,
                      request_timeout=120.0,
                      temperature=ollama_temperature,
                      base_url=env.OLLAMA_HOST)


@st.cache_resource(show_spinner=False)
def load_data() -> BaseIndex:
    logger.info('Loading data from collection: %s', chroma_collection.name)
    with st.spinner(text='Loading the index...'):
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection, ssl=False)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model,
        )


index = load_data()
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT,
    similarity_top_k=top_k,
    verbose=env.DEBUG)

# Initialize the chat message history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content='Ask me a question!'
        )
    ]

for message in st.session_state.messages:
    with st.chat_message(message.role):
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
    msg = ChatMessage(role=MessageRole.USER, content=prompt)
    st.session_state.messages.append(msg)
    with st.chat_message(msg.role):
        st.write(msg.content)

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
            del st.session_state.messages[-1]  # weird dup bug

            response = chat_engine.chat(
                message=prompt,
                chat_history=st.session_state.messages)

            if env.PDF_INCLUDE_SOURCE:
                meta_info = []
                for node in response.source_nodes:
                    metadata = node.node.metadata
                    text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
                    logger.debug(f'Text: {text_fmt}')
                    logger.debug(f"Metadata:\t {node.node.metadata}")
                    logger.debug(f"Score:\t {node.score:.3f}")
                    if 'url_src' in metadata:
                        meta_info.append(f'* {metadata["url_src"]}')
                    else:
                        file_name = basename(metadata.get('file_name', ''))
                        is_pdf = '.pdf' in file_name.lower()
                        page_label = metadata['page_label'] if 'page_label' in metadata else None
                        if page_label and is_pdf:
                            meta_info.append(f'* [{file_name}, page {page_label}]({env.PDF_SOURCE_LINK_PREFIX}{file_name.replace(" ", "%20")}#page={page_label})')
                        elif is_pdf:
                            meta_info.append(f'* [{file_name}]({env.PDF_SOURCE_LINK_PREFIX}{file_name.replace(" ", "%20")})')
                meta_info = list(set(meta_info))
                st.write(response.response + '\n\n' + '\n'.join(meta_info))
            else:
                st.write(response.response)

            # logger.info('---CHAT HISTORY---')
            # for message in st.session_state.messages:
            #     logger.info(message)
            # logger.info('---END HISTORY---')

    # Button to trigger speechSynthesis
    st.button(':speaking_head_in_silhouette: Read Aloud',
              on_click=speak_output,
              kwargs=dict(data=response.response))

if st.session_state.speakOutput:
    generate_speech_js()
