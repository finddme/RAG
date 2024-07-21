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
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from transformers import pipeline
# from langchain import HuggingFacePipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

print("STARTTTTTTT RAAGGGGGG")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"  # Set the GPU 2 to use
os.environ["OPENAI_API_KEY"] =  "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"
openai.api_key =os.getenv("OPENAI_API_KEY")
# pdf_loader = PdfReader(f'/workspace/yib/KimGoo_namuwiki.pdf')
# excel_loader = DirectoryLoader('./Reports/', glob="**/*.txt")
# word_loader = DirectoryLoader('./Reports/', glob="**/*.docx")
embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model)
nlp = spacy.load("en_core_web_sm")

print("load vectorstore")
vectorstore = FAISS.load_local("./KIDA_faiss_index2", embeddings)
time.sleep(20)
memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history",return_messages=True,output_key='answer')
# with open('./KIDA_faiss_index.json','r',encoding='utf-8-sig') as f:
#     txt_summary = json.load(f)

# login("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")
# tokenizer = AutoTokenizer.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4")
# model = AutoModelForCausalLM.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4")

login("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")

# print("model load")
# tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
# model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
# time.sleep(20)

# print("PeftModel load")
# model_path= "/workspace/Fastchat/checkpoint-10"
# model = PeftModel.from_pretrained(model, model_path,torch_dtype=torch.float16)
# time.sleep(20)


tokenizer = AutoTokenizer.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4")
model = AutoModelForCausalLM.from_pretrained("LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4",device_map="auto",max_memory={1: "40GiB", 2: "40GiB", "cpu": "30GiB"})

# model.config.max_length = 512
time.sleep(20)
# model_path= "/workspace/DOLLY/LLAMA2_13b_6epoch"
model_path= "./LLAMA2_13b_total2"
model = PeftModel.from_pretrained(model, model_path,torch_dtype=torch.float16)

print("make pipeline")
llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=512,
    # device_map={'cuda':0}
)

llm = HuggingFacePipeline(
    pipeline=llm_pipeline,
    # model_kwargs={"temperature": 0.7, "max_length": 512},
)



def get_embedding(query, engine=embed_model) : 
    # res = openai.Embedding.create(input=query,engine=engine)['data'][0]['embedding']
    from openai import OpenAI
    client = OpenAI()
    res= client.embeddings.create(input = query, model=engine).data[0].embedding
    return res

def return_answer_candidate_for_fliter(query):
    query_embedding = get_embedding(
        query,
        engine='text-embedding-ada-002'
    )


    # get relevant contexts (including the questions)
    # res = vectorstore.query(query_embedding, top_k=5, include_metadata=True)
    res = vectorstore.similarity_search_by_vector(query_embedding, k=10, fetch_k=40)
    res_list={'source':[]}
    #'/workspace/yib/KIDA/kida_data_txt/A-20-1.pdf.txt',
    for i in res:
        res_list['source'].append(i.metadata['source'])
    print("res_listres_listres_listres_list",len(res_list['source']),res_list)
    return res_list

def return_answer_candidate(query):
    query_embedding = get_embedding(
        query,
        engine='text-embedding-ada-002'
    )


    # get relevant contexts (including the questions)
    # res = vectorstore.query(query_embedding, top_k=5, include_metadata=True)
    res = vectorstore.similarity_search_by_vector(query_embedding, k=3, fetch_k=10)
    return res

def return_candidate_docs(query):
    res=return_answer_candidate_for_fliter(query)
    # retriever =vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3,"filter":res} 
                                                )
    docs = retriever.get_relevant_documents(query)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    
    return reordered_docs

def retriever_qa_doc(user_message,memory,vectorstore=vectorstore,llm=llm):
    reordered_docs=return_candidate_docs(user_message)
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    # if "표" in user_message or "차트" in user_message or "챠트" in user_message:
    #     stuff_prompt_override="""
    #         Below is an instruction(question) that describes a task. Write a response that appropriately completes the request, referring to given Context.
    #         When generating responses, it is crucial to adhere to the following conditions.
    #         - Do not generate new question. You must respond only to the given instruction.
    #         - You should never repeat the same sentence.
    #         - Do not repeat the instructions in your response.
    #         Responses related to specific numerical values within the answer must be maintained in JSON format. An example of the JSON format is as follows:
    #         {"인도태평양 사령부":"22명", "호주 해공군 전력":"28명"}

    #         ### instruction: {query}
    #         ### context :  {context}
    #         ### response :
    #     """     
    # else: 
    
    stuff_prompt_override = """
        Below is an instruction(question) that describes a task. Write a response that appropriately completes the request, referring to given Context.
        When generating responses, it is crucial to adhere to the following conditions.
        - Do not generate new question. You must respond only to the given instruction.
        - You should never repeat the same sentence.
        - Do not repeat the instructions in your response.

        Refer to the previous answer when answering.
        You should answer while referencing the previous answer.
        ## Previous answer : """+ memory + """

        ### instruction: {query}
        ### context :  {context}
        ### response :
        """
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    ) 
    result=chain.run(input_documents=reordered_docs, query=user_message)
    return result,reordered_docs

def retriever_qa_llmchain(user_message,vectorstore=vectorstore,llm=llm):
    # prompt = PromptTemplate.from_template(
    #     "Summarize the main themes in these retrieved docs: {docs}"
    # )

    p=f""" Below is an instruction that describes a task. Write a response that appropriately completes the request, referring to Input.
    ### instruction : {user_message}"""
    prompt = PromptTemplate.from_template(p+"""
    ### context :  {docs}
    ### response : 
        """)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    res=return_answer_candidate(user_message)
    output = llm_chain(res)   
    result=output['text']
    reordered_docs=output['docs']
    return result,reordered_docs

def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"

    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.5,
    )

    translation = response.choices[0].message.content.strip()
    return translation
    
def make_memory(memory):
    try:
        mem=memory.load_memory_variables({})['chat_history'][2:4]
        print(type(mem),type(mem),type(mem),type(mem),type(mem),type(mem))
        # rr1=re.compile("(HumanMessage\(content=)(.*?)(, additional_kwargs)")
        # rr1=rr1.findall(mem)[0][1]
        # rr2=re.compile("(AIMessage\(content=)(.*?)(, additional_kwargs)")
        # rr2=rr2.findall(mem)[0][1]
        rr1=str(mem[0]).replace("HumanMessage(content=","").replace(")","").replace("content=","")
        rr2=str(mem[1]).replace("AIMessage(content=","").replace(")","").replace("content=","")
        with open("./load_memory_variables.txt", "a") as f:
            f.write(f"{mem}, {rr1}, {rr2}\n")
        memmem=f"Previous question: {rr1}, Previous answer: {rr2}"
    except Exception as e: memmem,rr1,rr2="" ,"",""
    return memmem,rr1,rr2

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
                return f"source file: {source.split('KIDA/')[1]}\n\n{f.readlines()[:50]}"
        except Exception as e: return ""

    def user(user_message, history):
        print("Type of use msg:",type(user_message))
        memmem,Previous_question,Previous_answer=make_memory(memory)
        user_message_for_input= user_message + f"({Previous_question})"
        if "자세히" in user_message or "상세히" in user_message:
            user_message_for_input= f"({Previous_question})"+ user_message 
        else: user_message_for_input= user_message +f"({Previous_question})"
        response,reordered_docs= retriever_qa_doc(user_message=user_message_for_input,memory=Previous_answer)
        '''
        count=0
        for i in range(5):
            if "?" in response and count <= 1:
                print("??????????????????")
                response,reordered_docs= retriever_qa_llmchain(user_message=user_message)
                count+=1
            elif "?" in response and count >1:
                response="관련 문서를 찾을 수 없습니다."
            else: 
                print("!!!!!!!!!!!!!!!!!")
                break
        '''

        response2=response.replace("### instruction:","").replace("### context:","").replace("### response:","")

        

        response_list=[]
        for idx,i in enumerate(response2.split("\n")):
            if i not in response_list and "?" not in i: 
                response_list.append(i)
            # if "?" in i: 
            #     if idx ==0: response_list.append("관련 문서를 찾을 수 없습니다.")
                break
        answer="\n".join(response_list)

        En_detect=re.compile("[a-zA-Z]+")
        answer_is_en=""" """+answer+""" """
        if len(En_detect.findall(answer_is_en)) >= 10 :
            print("translate_text translate_text translate_text translate_text translate_text")
            answer = translate_text(answer_is_en, "English", "Korean")

        with open("./source_list.txt", "a") as f:
            f.write(f"""{reordered_docs}\n""")

        if response != "관련 문서를 찾을 수 없습니다.":
            source11=check_source(reordered_docs[0].metadata['source'])
            source22=check_source(reordered_docs[1].metadata['source'])
            source33=check_source(reordered_docs[2].metadata['source'])
        else: 
            source11=""
            source22=""
            source33=""

        history.append((user_message, answer))
        memory.save_context({"input": user_message}, {"answer": answer})
        memory.save_context({"input": user_message}, {"answer": answer})

        with open("./daa_kida.txt", "a") as f:
            f.write(f"(user_message: {user_message_for_input}, answer: {answer})\n")

        return gr.update(value=""), history, source11, source22, source33
    def clear_func(memory=memory):
        lambda: None
        memory.clear()

    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot,source_view1,source_view2,source_view3], queue=False)
    # clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(clear_func, None, chatbot, queue=False)
    

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True,server_name = "0.0.0.0",server_port=7808,
                # enable_queue=False
                )