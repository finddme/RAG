import json
from sentence_transformers import util
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
# from langchain import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain import LLMChain
# from langchain import retrievers
import langchain
import tiktoken
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFDirectoryLoader
import os
import openai
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryBufferMemory
from langchain.document_loaders import DirectoryLoader

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key =os.getenv("OPENAI_API_KEY")


# pdf_loader = PdfReader(f'/workspace/yib/KimGoo_namuwiki.pdf')
# excel_loader = DirectoryLoader('./Reports/', glob=\"**/*.txt\")
# word_loader = DirectoryLoader('./Reports/', glob=\"**/*.docx\")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

loader = PyPDFDirectoryLoader("example_data/")

docs=loader.load()

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer = tiktoken.encoding_for_model("gpt-4") # "gpt-4"도 가능
def tiktoken_len(text):
    tokens = tokenizer.encode(
    text,
    disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=tiktoken_len,
                separators=["\n\n\n", "\n\n", "\n", " ", ""] # Text splitter separators are very very important !!!!
                )

print("make chunks")
chunks=[]
for idx,page in enumerate(docs):
    #if len(page['text']) < 200:
    # if page content is short we can skip
    # continue
    texts = text_splitter.split_text(page.page_content)
    chunks.extend([{
    # 'metadata': {'source': page.metadata['source'] + f'-{i}'},
    'metadata': {'source': page.metadata['source']},
    'page_content': texts[i],
    # 'chunk': i
    } for i in range(len(texts))])

print(f"Total chunks: {len(chunks)}")

print("make vectorstore")
for idx, i in enumerate(chunks):
    if idx% 100 ==0: print(f"make vectorstore: {idx}")
    if idx ==0:
        vectorstore = FAISS.from_texts(texts=[i['page_content']],metadatas=[i['metadata']], embedding=embeddings)
    else: vectorstore.add_texts(texts=[i['page_content']],metadatas=[i['metadata']], embedding=embeddings)

print("save vectorstore")
vectorstore.save_local("/workspace/yib/KIDA/KIDA_faiss_index")
