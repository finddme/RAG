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
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain import LLMChain
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

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.load_local("./kimgu_faiss_index", embeddings)

# memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,output_key='answer')
memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history",return_messages=True,output_key='answer')
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512)

llm = ChatOpenAI(temperature=0,model_name="gpt-4-0613")

def retriever_qa(user_message,history,memory=memory,vectorstore=vectorstore,llm=llm):
    
    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key =os.getenv("OPENAI_API_KEY")

    embed_model = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model=embed_model)

    def get_embedding(query) : 
        res = openai.Embedding.create(
        input=query,
        engine=embed_model
        )['data'][0]['embedding']
        return res
    '''
    def return_answer_candidate(query):
        query_embedding = get_embedding(
        query
        )
        res = vectorstore.similarity_search_by_vector(query_embedding, k=3)
        return res
    '''
    def create_prompt(query,history):
        # print("memmemmemmemmemmemmemmemmemmemmemmemmemmemmemmem",mem)
        # res = return_answer_candidate(query)
        # memmem=make_memory(history)

        prompt_template = """
        Use the following pieces of context to answer the question at the end. Return a accurate answer based on the context.
        You must respond with a complete sentence of about 40 characters. You must answer with a complete sentence.
        Refer to the previous conversation(Previous question and Previous answer) when answering.
        You should answer while referencing the previous conversation.
        You are 김구.
        You must answer as if you are 김구, a middle-aged man, narrating his own story.

        # Previous Conversation : """ + history + """

        # Context :  {context}

        # Question: {question}

        # Answer : 

        """

        return prompt_template
    '''
    def create_prompt(query,history):
        res = return_answer_candidate(query)
        memmem=make_memory(history)
        
        prompt_template = """
        Use the following pieces of context to answer the question at the end. Return a accurate answer based on the context.
        You must respond with a complete sentence of about 40 characters. You must answer with a complete sentence.
        You must answer in Korean Language. Do not use any other languages except Korean.
        Refer to the previous conversation(Previous question and Previous answer) when answering.
        You should answer while referencing the previous conversation.
        You are 김구.
        You must act like 김구, a middle-aged man, narrating his own story.


        # Previous Conversation : """ + memmem + """

        # Context :  """ + str(res[0].page_content) + """ \n """ + str(res[1].page_content) + """  \n """ + str(res[2].page_content) + """ 

        # Question: {question}

        # Answer :

        """

        return prompt_template,memmem
    '''
    prompt_template= create_prompt(query=user_message,history=history)
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    chain_type_kwargs = {"prompt": QA_PROMPT}

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    
    result = qa({"query": user_message})

    '''
    prompt_template,memmem= create_prompt(user_message,history)
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question"]
    )

    h_QA_PROMPT = PromptTemplate(
        template="{question}", input_variables=["question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=QA_PROMPT)
    human_message_prompt = HumanMessagePromptTemplate(prompt=h_QA_PROMPT)
    CHAT_PROMPT = ChatPromptTemplate.from_messages([system_message_prompt, 
                                                    human_message_prompt,
                                                    # MessagesPlaceholder(variable_name="chat_history"),
                                                    ])

    chain_type_kwargs = {"prompt": CHAT_PROMPT}
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k':1})

    # memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,output_key='answer')
    # memory = ConversationBufferWindowMemory(k=6,memory_key="chat_history",return_messages=True,output_key='answer')
    # qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.8), vectorstore.as_retriever(),qa_prompt=QA_PROMPT)
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    
    qa = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", 
                                                retriever=retriever, return_source_documents=True, 
                                                condense_question_prompt=CHAT_PROMPT,
                                                # get_chat_history=lambda h : h,
                                                # memory=memory,
                                                verbose=True)
    
    result = qa({"question": user_message,
                "chat_history":history
                # "chat_history": memory.load_memory_variables({})['chat_history']
                })
    '''

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


def verification(user_message,answer,vectorstore=vectorstore,llm=llm):
    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key =os.getenv("OPENAI_API_KEY")

    embed_model = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    def get_embedding(query, engine='text-embedding-ada-002') : 
        res = openai.Embedding.create(
        input=query,
        engine=engine
        )
        return res['data'][0]['embedding']

    def return_answer_candidate(query):
        embed_model = "text-embedding-ada-002"
        query_embedding = get_embedding(
        query,
        engine=embed_model
        )
        res = vectorstore.similarity_search_by_vector(query_embedding, k=3)

        return res

    def create_prompt(query,answer):
        '''
        res = return_answer_candidate(query)

        prompt_template = """
        Is the provided answer accurate based on the information in the context?
        Is the provided answer appropriate to the core content of the question?
        If both conditions are met, say "Yes". If not, say "No".

        # Context :  {context} """ + str(res[0].page_content) + """ 

        # Question: {question}

        # Answer : """ + answer + """ 

        """
        '''
        prompt_template = """
        Is the provided answer accurate based on the information in the context?
        Is the provided answer appropriate to the core content of the question?
        If both conditions are met, say "Yes". If not, say "No".

        # Context :  {context} 

        # Question: {question}

        # Answer : """ + answer + """ 

        """
        return prompt_template

    
    prompt_template= create_prompt(user_message,answer)
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    chain_type_kwargs = {"prompt": QA_PROMPT}

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

    
    result = qa({"query": user_message})

    
    return result

def make_memory(memory):
    try:
        mem=str(memory.load_memory_variables({})['chat_history'][2:4])
        rr1=re.compile("(HumanMessage\(content=)(.*?)(, additional_kwargs)")
        rr1=rr1.findall(mem)[0][1]
        rr2=re.compile("(AIMessage\(content=)(.*?)(, additional_kwargs)")
        rr2=rr2.findall(mem)[0][1]
        memmem=f"Previous question: {rr1}, Previous answer: {rr2}"
    except Exception as e: memmem=""
    return memmem

import gradio as gr
import re
with gr.Blocks() as demo:
    gr.Markdown("## 김구 ChatBot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")


    def user(user_message, history):
        print("Type of use msg:",type(user_message))

        user_message_for_input=user_message.replace("선생님","")
        memmem=make_memory(memory)
        if "이전" in user_message:
            user_message_for_input += f"(이전 대화: {memmem})"

        response= retriever_qa(user_message=user_message_for_input,history=memmem)

        answer=response['result']

        En_detect=re.compile("[a-zA-Z]+")
        if En_detect.match(answer):
            print("translate_text")
            answer = translate_text(answer, "English", "Korean")
        if "이전" not in user_message:
            verification_result=verification(user_message_for_input,answer)['result']
            print(verification_result)
            
            if verification_result =="No" or "제공하지" in answer or "죄송" in answer or "찾을 수 없": 
                print("answer_again")
                answer=retriever_qa(user_message=user_message_for_input,history=memmem)['result']

        En_detect=re.compile("[a-zA-Z]+")
        if En_detect.match(answer):
            print("translate_text")
            answer = translate_text(answer, "English", "Korean")

        answer2=answer.replace("김구 선생님은","나는").replace("김구는","나는").replace("김구선생님은","나는")
        answer2=answer2.replace("김구 선생님의","나의").replace("김구선생님의","나의").replace("김구의","나의")
        answer2=answer2.replace("김구 선생님이","내가").replace("김구선생님이","내가").replace("김구가","내가")
        
        history.append((user_message, answer2))
        memory.save_context({"input": user_message}, {"answer": answer})
        memory.save_context({"input": user_message}, {"answer": answer})
        
        with open("./daa.txt", "a") as f:
            f.write(f"{str(memory.load_memory_variables({})['chat_history'])}\n")

        return gr.update(value=""), history

    def clear_func(memory=memory):
        lambda: None
        memory.clear()

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(clear_func, None, chatbot, queue=False)
    

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True,server_name = "0.0.0.0",server_port=8879,enable_queue=False)