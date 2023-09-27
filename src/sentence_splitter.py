# Python built-in module
import os
import time
import json

# Python installed module
import tiktoken
import langchain
from spacy.lang.en import English


class SentencizerSplitter(object):
    def __init__(self, config_dict):
        self.total_tokens = config_dict["embedding"]["total_tokens"]
        self.approx_total_doc_tokens = config_dict["sentence_splitter"]["approx_total_doc_tokens"]
        self.tolerance_limit_tokens = config_dict["sentence_splitter"]["tolerance_limit_tokens"]
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.encoding = tiktoken.encoding_for_model(config_dict["embedding"]["model_name"])
        
    def create_documents(self, content):
        nlp_sentences = list()
        nlp_sentences_docs = list()
        token_sum = 0
        str_sum = ""
        nlp_docs = self.nlp(content)
        for sent in nlp_docs.sents:
            sent_total_tokens = len(self.encoding.encode(sent.text))
            if sent_total_tokens + token_sum >= self.approx_total_doc_tokens + self.tolerance_limit_tokens:
                nlp_sentences.append(str_sum)
                str_sum = sent.text
                token_sum = sent_total_tokens
            else:
                str_sum += sent.text
                token_sum += sent_total_tokens
        if str_sum:
            nlp_sentences.append(str_sum)
        for chunk in nlp_sentences:
            nlp_sentences_docs.append(langchain.schema.document.Document(page_content=chunk))
        return nlp_sentences_docs