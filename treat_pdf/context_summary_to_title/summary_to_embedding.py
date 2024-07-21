from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from PyPDF2 import PdfReader
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Chroma
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
import json, spacy
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
import os
import openai
from langchain.memory import ConversationBufferWindowMemory,ConversationSummaryBufferMemory
from langchain.document_loaders import DirectoryLoader

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key =os.getenv("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model)

index_name="./conflict_specific_faiss_index"
specific_vectorstore = FAISS.load_local(index_name, embeddings)
vc=specific_vectorstore.docstore._dict

def get_embedding(query, engine='text-embedding-ada-002') : 
    res = openai.Embedding.create(
    input=query,
    engine=engine
    )
    return res['data'][0]['embedding']


result={}
for idx,(k,v) in enumerate(vc.items()):
    if idx %100==0:print(idx)
    summary = v.metadata['source'].split('/')[1]
    summary_embedding=get_embedding(
                                    summary,
                                    engine=embed_model
                                    )
    result[summary]=summary_embedding

import json
with open(f"{index_name}.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(result,ensure_ascii=False))