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
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel 
import torch
import json, spacy
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
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from transformers import pipeline
from langchain import HuggingFacePipeline
import time

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key =os.getenv("OPENAI_API_KEY")
# pdf_loader = PdfReader(f'/workspace/yib/KimGoo_namuwiki.pdf')
# excel_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
# word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")
embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model)
nlp = spacy.load("en_core_web_sm")

print("load vectorstore")
vectorstore = FAISS.load_local("./KIDA_faiss_index", embeddings)
time.sleep(20)

# with open('./KIDA_faiss_index.json','r',encoding='utf-8-sig') as f:
#     txt_summary = json.load(f)

# login("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")
# tokenizer = AutoTokenizer.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4")
# model = AutoModelForCausalLM.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4")

login("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")

print("model load")
tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
time.sleep(20)

print("PeftModel load")
model_path= "/workspace/Fastchat/checkpoint-10"
model = PeftModel.from_pretrained(model, model_path,torch_dtype=torch.float16)
time.sleep(20)

print("make pipeline")
llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=128,
    device_map={'cuda':0}
)

llm = HuggingFacePipeline(
    pipeline=llm_pipeline,
    # model_kwargs={"temperature": 0.7, "max_length": 512},
)



def get_embedding(query, engine=embed_model) : 
    res = openai.Embedding.create(input=query,engine=engine)['data'][0]['embedding']
    return res



def get_summary_response(text):
    gpt_prompt=f"""Provide a text separated by three quotation marks. 
                Summarize it into a sentence of around 30 characters.
                The summarized sentence should end with '~에 관한 갈등'

                \"\"\"{text}\"\"\"
                """
    message=[{"role": "user", "content": gpt_prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages = message,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )['choices'][0]['message']['content']
   
    return response

def retriever_qa(user_message,vectorstore=vectorstore,llm=llm):
    prompt_template = """
    Below is an instruction that describes a task. Write a response that appropriately completes the request, referring to Input.
    ### Input :  {context}

    ### Instruction: {question}

    ### Response : 

    """

    # prompt_template= create_prompt(user_message,answer)
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    chain_type_kwargs = {"prompt": QA_PROMPT}

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

    
    result = qa({"query": user_message})

    
    return result


def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    translation = response.choices[0].message.content.strip()
    return translation
    


import gradio as gr
import re
with gr.Blocks() as demo:
    gr.Markdown("""
    ## KIDA ChatBot

    """)
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=0.40):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter msg and press enter",
            )
            # .style(container=False) 
        with gr.Column(scale=0.20):
            source_view1 = gr.Textbox(label='source1',value=0)
        with gr.Column(scale=0.20):
            source_view2 = gr.Textbox(label='source2',value=0)
        with gr.Column(scale=0.20):
            source_view3 = gr.Textbox(label='source3',value=0)
    clear = gr.Button("Clear")
    chat_history = []
    
    def check_source(source):
        print("check source check source check source")
        try:
            # if ".pdf" in source:
            #     return f"source file: {source.split('/')[1]}"
            # else:
            # source=source.split(".txt")[0]+".txt"
            with open(f"./{source.split('KIDA/')[1]}", 'r') as f:
                return f"source file: {source.split('KIDA/')[1]}\n\n{f.readlines()}"
        except Exception as e: return ""

    def user(user_message, history):
        print("Type of use msg:",type(user_message))
        response= retriever_qa(user_message=user_message)

        with open("./source_list.txt", "a") as f:
          f.write(f"""
                    {response['source_documents']}\n
                    """)

        if len(response['source_documents']) ==0:
            source11=""
            source22="" 
            source33=""          
        elif len(response['source_documents']) >=1 and len(response['source_documents']) <2:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=""
            source33=""
        elif len(response['source_documents']) >=2 and len(response['source_documents']) <3:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=check_source(response['source_documents'][1].metadata['source'])
            source33=""
        elif len(response['source_documents']) >=3:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=check_source(response['source_documents'][1].metadata['source'])
            source33=check_source(response['source_documents'][2].metadata['source'])



        answer=response["result"]

        history.append((user_message, answer))

        with open("./daa_kida.txt", "a") as f:
          f.write(f"(user_message: {user_message}, answer: {answer})\n")

        return gr.update(value=""), history, source11, source22, source33

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot,source_view1,source_view2,source_view3], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True,server_name = "0.0.0.0",server_port=7808,
                # enable_queue=False
                )