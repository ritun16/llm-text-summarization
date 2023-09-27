# Python built-in module
import os
import time
import json
import traceback

# Python installed module
import openai
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Python user defined module
import prompts


class COD(object):
    '''This class implements the Chain-Of-Density summarization'''
    
    def __init__(self, config_dict):
        self.chat_4_llm = ChatOpenAI(model=config_dict["cod"]["model_name"],
                                   temperature=config_dict["cod"]["temperature"],
                                   max_tokens=config_dict["cod"]["max_tokens"],
                                   model_kwargs={"top_p": config_dict["cod"]["top_p"],
                                                 "presence_penalty": config_dict["cod"]["presence_penalty"],
                                                 "frequency_penalty": config_dict["cod"]["frequency_penalty"]})
        
        self.chat_turbo_llm = ChatOpenAI(model=config_dict["kw_extract"]["model_name"],
                                   temperature=config_dict["kw_extract"]["temperature"],
                                   max_tokens=config_dict["kw_extract"]["max_tokens"],
                                   model_kwargs={"top_p": config_dict["kw_extract"]["top_p"],
                                                 "presence_penalty": config_dict["kw_extract"]["presence_penalty"],
                                                 "frequency_penalty": config_dict["kw_extract"]["frequency_penalty"]})
    
    def __call__(self, text_content):
        try:
            start_time = time.time()
            print("[INFO] The Chain Of Density summarization started...")
            kw_extract_messages = [
                                      HumanMessage(content=prompts.KW_EXTRACT_SYSTEM_PROMPT.format(text_chunk=text_content))
                                  ]
            
            cod_messages = [
                                SystemMessage(content=prompts.COD_SYSTEM_PROMPT),
                                HumanMessage(content="Here is the input text for you to summarize using the 'Missing_Entities' and 'Denser_Summary' approach:\n\n{}".format(text_content))
                           ]
            
            with get_openai_callback() as openai_cb:
                kw_response = self.chat_turbo_llm(kw_extract_messages)
                cod_response = self.chat_4_llm(cod_messages)
            
            kw_output = kw_response.content.split(", ")
            output = cod_response.content
            
            try:
                output_dict = json.loads(output.replace("\n", ""))
                summary = output_dict[-1]['Denser_Summary']
                print("[INFO] The Chain Of Density summarization done!")
                end_time = time.time()
                return {"summary": summary,
                        "keywords": kw_output,
                        "metadata": {"total_tokens": openai_cb.total_tokens,
                                     "total_cost": round(openai_cb.total_cost, 3),
                                     "total_time": round((end_time-start_time), 2)}}
            except json.JSONDecodeError:
                print("[ERROR] The output JSON is not valid of the COD prompt response. LLM Output:\n\n{}\n\n".format(output))
                return
            except KeyError:
                print("[ERROR] The COD output JSON is missing the key `Denser_Summary`. Valid keys are `Missing_Entities` & `Denser_Summary`. LLM Output:\n\n{}\n\n".format(output))
                return
        except Exception as error:
            print("[ERROR] Some error happend in COD. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return