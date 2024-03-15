# ML Playground

This is a personal playground for experimenting with Machine
Learning tools, including Retrieval-Augmented Generation (RAG)
using either [LangChain](https://www.langchain.com/) or
[LlamaIndex](https://www.llamaindex.ai/).
This project is not intended for general use.

## Prerequisites

* Install and run [Ollama](https://ollama.com/)
* Download the necessary models (e.g. llama3.2)

## Installation

Clone the repository:

```shell
git clone https://github.com/wojciechpolak/ml-playground.git
```

Navigate to the project directory and set up a virtual environment:

```shell
cd ml-playground
python3 -m venv venv
source venv/bin/activate
poetry install
```

### Setting Up Environment Variables

Configure environment variables by creating a `.env` file
in the root directory. Here is an example of the variables
you may need:

```ini
APP_TITLE=My AI Assistant
CHROMA_HOST=
CHROMA_PATH=
CHROMA_PORT=
COLLECTION_NAME=default
COLLECTION_RESET=
DEBUG=
HF_EMBEDDING=
INDEXER_LOADER=dir|pdf|obsidian
INDEXER_SPLIT_DOCS=
LLAMA_INDEX_CACHE_DIR=./run/cache
NLTK_DATA=./run/nltk_data
OLLAMA_EMBEDDING=all-minilm
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TEMPERATURE=0.75
PDF_INCLUDE_SOURCE=
PDF_SOURCE_LINK_PREFIX=
QE=
SENTENCE_TRANSFORMERS_HOME=./run/cache
TOKENIZERS_PARALLELISM=true
TOP_K=5
WHISPER_LANG=
WL_BEARER_TOKEN=
WL_COOKIE_TOKEN=
WL_DEPTH_LIMIT=0
```

## RAG Chat

This project includes RAG implementations for answering questions from
personal knowledge bases like Obsidian notes, leveraging LangChain or
LlamaIndex.

### Indexing

Choose between LangChain or LlamaIndex for indexing your knowledge
base:

```shell
python -m rag.ver_langchain.indexer ~/Obsidian/Notes/
python -m rag.ver_llamaindex.indexer ~/Obsidian/Notes/
```

To test the retriever:

```shell
python -m rag.ver_langchain.retriever "your query here"
```

### Embedding Models

Ollama Models:
* all-minilm
* mxbai-embed-large
* nomic-embed-text

HuggingFace Models:
* intfloat/multilingual-e5-small
* ipipan/silver-retriever-base-v1.1
* sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

### Running the Chat

Text-based chat version:

```shell
python -m rag.ver_langchain.chat_text
```

UI version using [Streamlit](https://streamlit.io/):

```shell
PYTHONPATH=. streamlit run rag/ver_langchain/chat_ui.py
```

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for more details.
