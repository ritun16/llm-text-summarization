COD_SYSTEM_PROMPT = """You will generate increasingly concise, entity-dense summaries of the above article. 
Repeat the following 2 steps 5 times. 
Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 
A missing entity is:
- relevant to the main story, 
- specific yet concise (5 words or fewer), 
- novel (not in the previous summary), 
- faithful (present in the article), 
- anywhere (can be located anywhere in the article).
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
Remember, use the exact same number of words for each summary.
Answer in valid JSON. The JSON should be a python list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""

KW_EXTRACT_SYSTEM_PROMPT = """You are an efficient key word detector. Your task is to extract only all the important key words and phrases without any duplicates from the below chunk of text.

Text: {text_chunk}

Think "step by step" to identify and all the important key words and pharses only and output should be comma seperated.
Important Keywords:"""

SEQUENCIAL_SUMMARY_PROMPT = """You are an expert text summarizer. Given the below text content and the important key words, write a concise but information loaded summary.

Text Content: {text_chunk}

Important Keywords: {key_words}

Think "step by step" how to utilize both the important keywords and text content to create a great concise summary.
Summary:"""

REDUCE_PROMPT = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary.
Final Summary:"""