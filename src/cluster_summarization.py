# Python built-in module
import os
import time
import json
import traceback

# Python installed module
import openai
import tiktoken
import langchain
import numpy as np
from sklearn.cluster import KMeans
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

# Python user defined module
import prompts
from map_reduce import MapReduce
from sentence_splitter import SentencizerSplitter


class ClusterBasedSummary(object):
    '''This class implements the clustered based summarization'''
    
    def __init__(self, config_dict):
        self.embeddings = OpenAIEmbeddings(model=config_dict["embedding"]["model_name"])
        self.embedding_chunk_size = config_dict["embedding"]["chunk_size"]
        self.num_clusters = config_dict["cluster_summarization"]["num_clusters"]
        self.k = config_dict["cluster_summarization"]["num_closest_points_per_cluster"]
        self.config_dict = config_dict
        
        self.text_splitter = SentencizerSplitter(self.config_dict)
        self.map_reduce_summarizer = MapReduce(config_dict)
        
    def __call__(self, text_content):
        try:
            with get_openai_callback() as openai_cb:
                start_time = time.time()
                print("[INFO] Cluster based summarization started...")
                print("[INFO] Text chunking started...")
                document_splits = self.text_splitter.create_documents(text_content)
                total_splits = len(document_splits)
                print("[INFO] Text chunking done!")
                print("[INFO] Text embedding started...")
                vectors = self.embeddings.embed_documents(texts=[x.page_content for x in document_splits], chunk_size=self.embedding_chunk_size)
                print("[INFO] Text embedding done!")
                print("[INFO] K-Means clustering started, this might take some time...")
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(vectors)
                print("[INFO] K-Means clustering done!")
                print("[INFO] Finding the closest points to each cluster center")
                closest_indices = []
                for i in range(self.num_clusters):
                    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                    closest_index = np.argsort(distances)[:self.k]
                    closest_indices.extend(closest_index)
                    selected_indices = sorted(closest_indices)
                sorted_documents = [document_splits[idx].page_content for idx in selected_indices]
                mr_result_dict = self.map_reduce_summarizer("\n".join(sorted_documents), redirect="cluster_summary")
                end_time = time.time()
                print("[INFO] Cluster based summarization done!")
                
            return {"summary": mr_result_dict["summary"],
                    "keywords": mr_result_dict["keywords"],
                    "metadata": {"total_tokens": openai_cb.total_tokens + mr_result_dict["metadata"]["total_tokens"],
                                 "total_cost": round(openai_cb.total_cost, 3) + mr_result_dict["metadata"]["total_cost"],
                                 "total_time": round((end_time-start_time), 2)}}
            
        except Exception as error:
            print("[ERROR] Some error happend in Map Reduce. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return