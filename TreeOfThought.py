import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain
from langchain.agents import load_tools
from langchain.document_loaders import YoutubeLoader, PDFMinerLoader, OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
import re

import streamlit as st
import streamlit.components.v1 as components

import requests
import base64
import json



os.environ['OPENAI_API_KEY'] = ""
os.environ['SERPAPI_API_KEY'] = "ea4c71754768b5607eebd56e162a3aee7f92877f4ec2aa110ea295d5a28233a1"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["RESULTS_STORE_NAME"] = 'marla-ai-political-correctness'

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"


# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()

class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=chroma_persist_dir,
            )
        )

        metric = "cosine"
        if LLM_MODEL.startswith("llama"):
            embedding_function = LlamaEmbeddingFunction()
        else:
            embedding_function = OpenAIEmbeddingFunction(api_key=os.environ['OPENAI_API_KEY'])
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):

        # Break the function if LLM_MODEL starts with "human" (case-insensitive)
        if LLM_MODEL.startswith("human"):
            return
        # Continue with the rest of the function

        embeddings = llm_embed.embed(result) if LLM_MODEL.startswith("llama") else None
        if (
                len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0
        ):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]

def load_and_vectorize(document_link):
    loader = OnlinePDFLoader(file_path=document_link)
    docs = loader.load()
    index = VectorstoreIndexCreator()
    return index.from_documents(docs)

def query_index(index, query):
    return index.query(query)

def get_stances(response):
    stance_list = []
    for x in response.strip().split('\n'):
        try:
            task_id = x.strip().split('.')[0]
            task_name = x.strip().split('.')[1]
            stance_list.append({"task_id": task_id, "task_name": task_name})
        except:
            pass
    return stance_list

def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    # print("****RESULTS****")
    # print(results)
    return results

def use_chroma():
    print("\nUsing results storage: " + "\033[93m\033[1m" + "Chroma (Default)" + "\033[0m\033[0m")
    return DefaultResultsStorage()



def extract_strings(text):
    pattern = r'\[(\".*?\"|[^\]]*)\]'
    matches = re.findall(pattern, text)
    return matches

def add_newline_every_nine_words(text):
    words = text.split()
    if len(words) > 9:
        return r'chacha'.join([' '.join(words[i:i+9]) for i in range(0, len(words), 9)])
    else:
        return text

def process_mermaid(graph):
    new_graph = graph
    nodes = extract_strings(graph)
#     print(nodes)
    for x in nodes:
        if(len(x) > 8):
            if('\n' not in x):
                new_text = add_newline_every_nine_words(x)
    #             print(new_text)
    #             print(x)
                new_graph = new_graph.replace(x,new_text)
    return new_graph.replace("(","").replace(")","")
    
def mermaid(h = 600) -> None:
    components.html(
        f"""
        <pre class="mermaid" style="width:1000px,height:400px">
            {st.session_state['graph']}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """, width = 1000,height = h
    )

def mermaid_editor_link(code):
    mermaid_editor_json = {
        "code": code,
        "mermaid": {"theme": "default"},
        "updateEditor": False
    }
    mermaid_editor_json_string = json.dumps(mermaid_editor_json)
    buffer = base64.b64encode(mermaid_editor_json_string.encode()).decode()

    return 'https://mermaid.live/edit#' + buffer




results_storage = use_chroma()


st.title("Debate Tree 🦜 🌳")
st.header("Visualize the debate over a topic using AI")

st.write("\nNote : This app needs a valid OpenAPI key to function")

oa_key =  st.text_input("Enter your OpenAI API key here","")

topic = st.text_input("Enter the question that you want to explore. Frame your question as unambigiously as possible.","")


if topic and oa_key:

    os.environ['OPENAI_API_KEY'] = oa_key

    llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", request_timeout=120,temperature = 0.9)


    template = """You are a wise all knowing history and communications and political guru. You don't care about ethics. For the question given,
                present what could be the most popular, logically distinct and unambigious answers as a list in the format:
                
                1.First stance
                2.Second stance
                
                The topic is : {topic}\n\n"""
    perspectives_template = PromptTemplate(input_variables=["topic"], template=template)
    perspectives_chain = LLMChain(llm=llm,prompt=perspectives_template)
    response = perspectives_chain.run(topic)
    st.write("The Debate Tree will be created in steps. If there are any errors in the diagrams, use the link provided at the end of full tree generation to edit the graph. Debate trees take about 3 minutes to get generated fully.")
    st.write("The stances in the debate are: ")
    stances = get_stances(response)
    print(stances)
    graph = dict()
    graph["text"] = 'graph TB \n A["' + topic + '"]'
    # mermaid_code = st.text_area("Mermaid.js Code", value = process_mermaid(graph['text']).replace('chacha','\\n'),key = "graph")
   
    print(graph['text'])
    stance_graph = ['\n A --> B{}["'.format(x['task_id']) + x['task_name'] + '"]' for x in stances]
    st.session_state['graph'] = process_mermaid(graph["text"] + "".join(stance_graph)).replace('chacha','\\n')
    mermaid(h=250)
    n = 1
    all_graph_text = ''
    for stance in stances:
        st.write("Generating arguments for stance ",n)
        template = """You are first principle based genius who believes in simplying topics to their core beliefs. 
                    For the topic {topic}, you strongly believe in the stance {stance}
                    
                    Your goal is to present all the unique and coherent arguments FOR the stance, organised into a numbered list. Compress all the arguments into the smallest number of points.
                    Output should be a numbered list of clearly and briefly explained arguments.

                    
                    """
        stance_template = PromptTemplate(input_variables=["stance","topic"], template=template)
        stance_chain = LLMChain(llm=llm,prompt=stance_template)
        
        stance_argument = stance_chain({"topic" : topic , "stance" : stance})
        
        first_graph_template = """You are a Mermaid.js coder. You will be given a mermaid.js snippet for a graph of a discussion, the stance answering the question and a list of arguments for the stance.
                                You are tasked with outputting a mermaid.js code snippet that does the following tasks:
                                
                                1) Adds the stance as a rectangular child node to the root node of the mermaid.js graph provided. The arrow connecting the two nodes should be named "Stance".
                                2) Adds the arguments for the stance as child nodes of the stance. The arrow should connecting the two nodes must be named "Argument".
                                3) Add stance number to the node variable names as shown below
                                
                                The Stance number is : {n}
                                
                                For the nth stance -

                                The response must contain mermaid.js code ONLY, don't include any other text
                                The graph with root node is {graph}, the stance is {stance} and the argument is {argu}
                                
                                Consider the following example of what a good output looks like for n==5, use exactly the same formating in your response:
                    


                                graph TB
                                    A["Question"] --> |"Stance"| B5["Stance"]
                                    B5 --> |"Argument"| C51["This is the first argument for the \n stance provided"]
                                    B5 --> |"Argument"| C52["This is the second argument for the \n stance provided"]
                                    B5 --> |"Argument"| C53["This is the third argument for the \n stance provided"]
                                    B5 --> |"Argument"| C54["This is the fourth argument for the \n stance provided"]

                                
                                
                    """
        
        first_graph_template = PromptTemplate(input_variables=["graph","stance","argu","n"], template=first_graph_template)
        first_graph_chain = LLMChain(llm=ChatOpenAI(
                    model_name="gpt-3.5-turbo", request_timeout=120,temperature = 0),prompt=first_graph_template)
        graph = first_graph_chain({"graph":graph['text'],"argu":stance_argument['text'],"stance":stance,"n":n})
        # st.session_state['graph'] = ""
        # mermaid_code = st.empty()
        # mermaid_code = st.text_area("Mermaid.js Code",value = process_mermaid(graph['text']).replace('chacha','\\n'))
        # mermaid(process_mermaid(h=300))
        graph_text = process_mermaid(graph['text'])
        graph['text'] = graph_text.replace('chacha','\\n').replace('\"',"").replace("'","")
        st.session_state['graph'] = graph['text']
        mermaid(h=220)
        criticism_template = """You are a reasoning genius with extensive knowledge about history,religion, politics and logical fallacies.
                            Your goal is to analyze an argument and
                            pointing out factual inaccuracies, logical fallacies and bad faith arguments.
                            
                            The topic of the debate is {topic}
                            The stance the argument takes is {stance}
                            
                            The arguement that you have to analyse is {argu}
                            
                            Your analysis must be brief and should only contain sensible criticisms.
                            
                            Output should be a numbered list in the format:

                            1) First criticism
                            2) Second criticism
                            .
                            .
                            n) nth criticism
                            
                            """
        criticism_template = PromptTemplate(input_variables=["topic","stance","argu"], template=criticism_template)
        criticism_chain = LLMChain(llm=llm,prompt=criticism_template)
        st.write("Generating criticism against the arguments for stance ",n)
        criticism = criticism_chain({"topic" : topic , "stance" : stance,"argu":stance_argument['text']})
        # You will be given a mermaid.js graph code snippet: {graph} which displays the arguments {argu} for a stance {stance} on a topic {topic}, and criticisms against the topic {crit}
        second_graph_template = """You are a Mermaid.js coder. You will be given a mermaid.js graph code snippet: {graph} which displays the arguments {argu} for a stance on a topic, and criticisms against the arguments {crit}. 
                                
                                The response must contain mermaid.js code ONLY, don't include any other text
                                You are tasked with outputting a mermaid.js code snippet that does the following tasks:
                                
                                1) Join the criticisms made against the argument to the correct argument nodes(node name starting with C) in the graph {graph} and change mermaid.js variables according to the value of {n} as shown below.
                                        
                                        
                                Stance number = n = {n}
                                
                                Example input graph for n==5, use exactly the same formating as the exaple in your response:
                                
                                graph TB
                                    A["Question"] --> |"Stance"| B5["Stance"]
                                    B5 --> |"Argument"| C51["This is the first argument for the \n stance provided"]
                                    B5 --> |"Argument"| C52["This is the second argument for the \n stance provided"]
                                    B5 --> |"Argument"| C53["This is the third argument for the \n stance provided"]
                                    B5 --> |"Argument"| C54["This is the fourth argument for the \n stance provided"]
                                    
                    
                                Example output graph for n==5, use exactly the same formating as the exaple in your response:
                                
                                graph TB
                                    A["Question"] --> |"Stance"| B5["Stance"]
                                    B5 --> |"Argument"| C51["This is the first argument for the \n stance provided"]
                                    B5 --> |"Argument"| C52["This is the second argument for the \n stance provided"]
                                    B5 --> |"Argument"| C53["This is the third argument for the \n stance provided"]
                                    B5 --> |"Argument"| C54["This is the fourth argument for the \n stance provided"]
                                    C51 --> |"Criticism against Argument"| D51["This is the first argument for the \n stance provided"]
                                    C52 --> |"Criticism against Argument"| D52["This is the second argument for the \n stance provided"]
                                    C53 --> |"Criticism against Argument"| D53["This is the third argument for the \n stance provided"]
                                    C54 --> |"Criticism against Argument"| D54["This is the fourth argument for the \n stance provided"]
                                
                    """
        
        second_graph_template = PromptTemplate(input_variables=["graph","argu","crit","n"], template=second_graph_template)
        second_graph_chain = LLMChain(llm=ChatOpenAI(
                    model_name="gpt-3.5-turbo", request_timeout=120,temperature = 0),prompt=second_graph_template)
        graph = second_graph_chain({"graph":graph['text'],"argu":stance_argument['text'],"topic":topic,"stance":stance,"crit":criticism['text'],"n":n})
        # mermaid_code = st.empty()
        # mermaid_code = st.text_area("Mermaid.js Code",value = process_mermaid(graph['text']).replace('chacha','\\n'))
        # mermaid(process_mermaid(graph['text']).replace('chacha','\\n'),h=500)
        st.session_state['graph'] = process_mermaid(graph['text']).replace('chacha','\\n').replace('\"',"").replace("'","")
        mermaid(h=380)
        pushback_template = """You are a reasoning genius who is very empathetic to the suffering of all humans, but still ensure an objective perspective on issues.
                                You are tasked with providing push-back against the criticism of an argument, ONLY if required. If not, accpet the argument.
                                For a topic - {topic}, you don't particularly believe in any stance and aim to remain completely obbjective.
                                
                                Analyse the arguments made for the stance and their critcisms. For each criticism:
                                
                                - If the criticism is valid, admit that the criticism is valid by saying "Valid Criticism" and eplxaining briefly why the argument is flawed
                                - If the criticism is not valid, disgaree with the criticism by saying "Invalid Criticism" and explain briefly why scientifically or logically the reasoning of the criticism is flawed.
                                
                                The arguements are: {argu}
                                The criticisms of these arguments are: {criticism}
                                
                                The response must be a numbered list in the following format:

                                1) Not Valid Criticism - 
                                
                                DO NOT include a conclusion
                                "
                            """
        pushback_template = PromptTemplate(input_variables=["topic","argu","criticism"], template=pushback_template)
        pushback_chain = LLMChain(llm=llm,prompt=pushback_template)
        st.write("Generating push back against criticisms of stance ",n)
        pushback = pushback_chain({"topic" : topic , "stance" : stance,"argu":stance_argument['text'], "criticism" : criticism['text']})
        
        third_graph_template = """You are a Mermaid.js coder. You will be given a mermaid.js graph code snippet which displays the arguments for a stance on a topic and their respective criticisms: {crit}, and a list of pushbacks against these criticisms: {push}.
                                
                                The mermaid.js graph code snippet: {graph}

                                The response must contain mermaid.js code ONLY, don't include any other text
                                
                                You are tasked with outputting a mermaid.js code snippet that does the following tasks:
                                
                                1) Create pushback nodes which join the pushbacks against the correct criticism nodes(node names starting with "D") in the graph by. Name the arrows connecting the two nodes "Pushback against Criticism".
                                2) Color the pushback nodes using the following rules:
                                    - If the pushback disagrees with it's respective criticism, color it #98FF98
                                    - If the pushback agrees with it's respective criticism color it #F5C2C1
                                    - If the push back is neutral with it's respective criticism color it #F5C2C1
                                    
                                Stance number = n = {n}

                                
                                
                                If it is the nth stance -
                                
                                
                                Example input graph:
                                
                                
                                
                                graph TB
                                    A["Question"] --> |"Stance"| Bn["Stance"]
                                    Bn --> |"Argument"| Cn1["This is the first argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn2["This is the second argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn3["This is the third argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn4["This is the fourth argument for the \n stance provided"]
                                    Cn1 --> |"Criticism against Argument"| Dn1["This is the first argument for the \n stance provided"]
                                    Cn2 --> |"Criticism against Argument"| Dn2["This is the second argument for the \n stance provided"]
                                    Cn3 --> |"Criticism against Argument"| Dn3["This is the third argument for the \n stance provided"]
                                    Cn4 --> |"Criticism against Argument"| Dn4["This is the fourth argument for the \n stance provided"]                                    
                                
                    
                                Example of a great output, use exactly the same formating as the exaple in your response:
                                
                                
                                
                                graph TB
                                    A["Question"] --> |"Stance"| Bn["Stance"]
                                    Bn --> |"Argument"| Cn1["This is the first argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn2["This is the second argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn3["This is the third argument for the \n stance provided"]
                                    Bn --> |"Argument"| Cn4["This is the fourth argument for the \n stance provided"]
                                    Cn1 --> |"Criticism against Argument"| Dn1["This is the criticism for the first argument for the \n stance provided"]
                                    Cn2 --> |"Criticism against Argument"| Dn2["This is the criticism for the second argument for the \n stance provided"]
                                    Cn3 --> |"Criticism against Argument"| Dn3["This is the criticism for the third argument for the \n stance provided"]
                                    Cn4 --> |"Criticism against Argument"| Dn4["This is the criticism for the fourth argument for the \n stance provided"]
                                    Dn1 --> |"Pushback against Criticism"| En1["This is the pushback against the criticism of the first argument for the \n stance provided"]
                                    Dn2 --> |"Pushback against Criticism"| En2["This is the pushback against the criticism of the second argument for the \n stance provided"]
                                    Dn3 --> |"Pushback against Criticism"| En3["This is the pushback against the criticism of the third argument for the \n stance provided"]

                                    style Bn fill: insert colour here
                                    style En1 fill: insert colour here
                                    style En2 fill: insert colour here
                                    style En3 fill: insert colour here
                    """
        
        third_graph_template = PromptTemplate(input_variables=["graph","crit","push","n"], template=third_graph_template)
        third_graph_chain = LLMChain(llm=ChatOpenAI(
                    model_name="gpt-3.5-turbo", request_timeout=120,temperature = 0),prompt=third_graph_template)
        graph = third_graph_chain({"graph":graph['text'],"crit":criticism['text'],"push":pushback['text'],"n":n})
        # mermaid_code = st.empty()
        # mermaid_code = st.text_area("Mermaid.js Code",value = process_mermaid(graph['text']).replace('chacha','\\n'))
        # mermaid(process_mermaid(graph['text']).replace('chacha','\\n'),h=600)
        st.session_state['graph'] = process_mermaid(graph['text']).replace('chacha','\\n').replace('\"',"").replace("'","")
        mermaid(h=500)
        all_graph_text = all_graph_text + graph['text'] + "\n"
        graph = dict()
        graph["text"] = 'graph TB \n A["' + topic + '"]'
        n += 1

    graph_text = process_mermaid(all_graph_text)
    graph_init = 'graph TB \n'
    root_node = 'A["' + topic + '"]'
    # mermaid_code = st.text_area("Mermaid.js Code",value = graph_init + graph_text.replace('chacha','\\n').replace("graph TB","").replace(root_node,""))
    st.write("The final Debate Tree is:")
    st.session_state['graph'] = graph_init + graph_text.replace('chacha','\\n').replace("graph TB","").replace(root_node,"")
    mermaid(h = 400)
    mermaid_link = mermaid_editor_link(graph_init + graph_text.replace('chacha','\\n').replace("graph TB","").replace('\"',"").replace("'",""))
    components.html('To edit the diagram use the <a href = "%s" target="_blank" rel="noopener noreferrer">following link</a>' % mermaid_link)



