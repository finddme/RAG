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
# pdf_loader = PdfReader(f'/workspace/yib/KimGoo_namuwiki.pdf')
# excel_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
# word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")
embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model)
nlp = spacy.load("en_core_web_sm")

specific_vectorstore = FAISS.load_local("./conflict_specific_faiss_index", embeddings)
common_vectorstore = FAISS.load_local("./conflict_common_faiss_index", embeddings)
vectorstore = FAISS.load_local("./conflict_faiss_index2", embeddings)

with open('./conflict_specific_faiss_index.jsonl','r',encoding='utf-8-sig') as f:
    txt_summary = json.load(f)

# memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history",return_messages=True,output_key='answer')
# llm = ChatOpenAI(temperature=0,model_name="gpt-4-0613")
llm = ChatOpenAI(temperature=0,model_name="gpt-4-1106-preview")

# memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history",return_messages=True,output_key='answer')
def get_embedding(query, engine=embed_model) : 
    res = openai.Embedding.create(input=query,engine=engine)['data'][0]['embedding']
    return res

def candidate_doc_search(query, engine=embed_model):
    query_embedding = get_embedding(query,engine=engine)
    
    sim_results={}
    for k,v in txt_summary.items():
        sim_result=float(util.pytorch_cos_sim(v, query_embedding)[0][0])
        if sim_result > 0.70: sim_results[k]=sim_result
    # sim_results1=dict(sorted(sim_results.items(), key=lambda x: x[1], reverse=True))
    # print(sim_results1)
    if len(list(sim_results.keys()))!= 0 :
        is_specific=True
        sim_results_sorted_key=sorted(sim_results,key=lambda x:sim_results[x], reverse=True)
        # print(sim_results_sorted_key)
        return sim_results_sorted_key
    else: return []

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

def retriever_qa(user_message,
                # history,
                # memory=memory,
                specific_vectorstore=specific_vectorstore,common_vectorstore=common_vectorstore,
                llm=llm,txt_summary=txt_summary):
    
    # def return_answer_candidate(query):
    #     embed_model = "text-embedding-ada-002"
    #     query_embedding = get_embedding(
    #                                     query,
    #                                     engine=embed_model
    #                                     )
    # def create_prompt(query):
    if len(user_message) > 150:
        user_message=get_summary_response(user_message[:5500]).replace("\"","").replace("\'","").replace(".","")
        
    # candidate_doc= candidate_doc_search(user_message)
    if len(candidate_doc_search(user_message)) !=0:
        print("specific_vectorstore specific_vectorstore specific_vectorstore")
        candidate_doc_filter={'source':candidate_doc_search(user_message)}
        # retriever = specific_vectorstore.as_retriever(search_type="mmr",
        #                                     # search_type="similarity", 
        #                                         search_kwargs={'k':5,'filter': candidate_doc_filter})

        retriever = specific_vectorstore.as_retriever(search_type="mmr",search_kwargs={'k':5})
        
        prompt_template="""You are specialized in summarizing and answering documents about a research on conflict resolving.
        You need to take a given document and return a very detailed summary of each document in the query language.
        You must answer in Korean Language. Do not use any other languages except Korean.
        Return a accurate answer based on the document.
        Certainly, your response needs to include information distinctly categorized under the following headings:
        - Causes of Conflict
        - Key Issues
        - Opinions of Stakeholders
        - Solutions

        # Question: {question}

        Here are the document: 
        # {context} 

        # Answer :
        - Causes of Conflict : 
        - Key Issues : 
        - Opinions of Stakeholders : 
        - Solutions : 
        """
    else: 
        print("common_vectorstore common_vectorstore common_vectorstore")
        retriever = common_vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':5})
        
        prompt_template="""You are specialized in summarizing and answering documents about a research on conflict resolving.
        You need to take a given document and return a very detailed summary of each document in the query language.
        You must answer in Korean Language. Do not use any other languages except Korean.
        Return a accurate answer based on the document.

        # Question: {question}

        Here are the document: 
        # {context} 

        # Answer :

        """
    
    common_triger=["갈등의 원인", "갈등의 해법","갈등의 관리"]
    for i in common_triger:
        if i in user_message:
            print("common_triger common_triger common_triger")
            retriever = common_vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':5})
        
            prompt_template="""You are specialized in summarizing and answering documents about a research on conflict resolving.
            You need to take a given document and return a very detailed summary of each document in the query language.
            You must answer in Korean Language. Do not use any other languages except Korean.
            Return a accurate answer based on the document.

            # Question: {question}

            Here are the document: 
            # {context} 

            # Answer :

            """

    # prompt_template= create_prompt(user_message,answer)
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    chain_type_kwargs = {"prompt": QA_PROMPT}

    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

    
    result = qa({"query": user_message})

    
    return result, candidate_doc_filter


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
    ## 행정연 ChatBot
    입력 예시:

    1. {갈등 원인}, {주요 쟁점} , {이해 당사자 의견} , {해결 방안} 요약:
        - O O 관련 갈등 사례의 {갈등 원인}, {주요 쟁점} , {이해 당사자 의견} , {해결 방안}을 구분해서 요약해줘.
        - "뉴스 기사" 위 뉴스 기사에 나타난 갈등 사례의 {갈등 원인}, {주요 쟁점} , {이해 당사자 의견} , {해결 방안}을 구분해서 요약해줘.
    2. 일반 질문: 갈등의 원인/갈등의 해법/ 갈등의 관리 방법에는 어떤 것들이 있어?
    3. 기타 질문: O O 갈등과 유사한 갈등 사례의 갈등 해결 방안을 요약해줘.
    """)
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=0.40):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter msg and press enter",
            ).style(container=False) 
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
            if ".pdf" in source:
                return f"source file: {source.split('/')[1]}"
            else:
                source=source.split(".txt")[0]+".txt"
                with open(f"./{source}", 'r') as f:
                    return f"source file: {source.split('/')[1]}\n\n{f.readlines()}"
        except Exception as e: return ""

    def user(user_message, history):
        print("Type of use msg:",type(user_message))
        response,candidate_doc_filter= retriever_qa(user_message=user_message)

        print(f"len(response['source_documents']): {len(response['source_documents'])}")

        with open("./source_list.txt", "a") as f:
          f.write(f"""
                    {response['source_documents']}\n
                    candidate_doc_filter: {candidate_doc_filter['source']}
                    """)

        if len(response['source_documents']) ==0:
            source11=check_source(candidate_doc_filter['source'][0])
            source22=check_source(candidate_doc_filter['source'][1]) 
            source33=check_source(candidate_doc_filter['source'][2])           
        elif len(response['source_documents']) >=1 and len(response['source_documents']) <2:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=check_source(candidate_doc_filter['source'][0])
            source33=check_source(candidate_doc_filter['source'][1])
        elif len(response['source_documents']) >=2 and len(response['source_documents']) <3:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=check_source(response['source_documents'][1].metadata['source'])
            source33=check_source(candidate_doc_filter['source'][0])
        elif len(response['source_documents']) >=3:
            source11=check_source(response['source_documents'][0].metadata['source'])
            source22=check_source(response['source_documents'][1].metadata['source'])
            source33=check_source(response['source_documents'][2].metadata['source'])



        answer=response["result"]

        history.append((user_message, answer))

        with open("./daa_con.txt", "a") as f:
          f.write(f"(user_message: {user_message}, answer: {answer})\n")

        return gr.update(value=""), history, source11, source22, source33

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot,source_view1,source_view2,source_view3], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True,server_name = "0.0.0.0",server_port=8787,enable_queue=False)