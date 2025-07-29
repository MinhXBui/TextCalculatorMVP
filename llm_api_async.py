import os
import asyncio
from asynciolimiter import Limiter
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity

def batch_prompt_processing (system_prompt, user_question, context_list):
  batch_request_list = []
  for each_content in context_list:
    prompt = f'Question: ({user_question}), Context: ({each_content})'
    batch_request_list.append([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
  return batch_request_list

async def multiple_llm_api_request (batch_request_list, model_configs):

  llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", **model_configs)
  results = await llm.abatch(batch_request_list)
  api_result_list = []
  for each_result in results:
    result_text_content = each_result.content
    api_result_list.append(result_text_content)
  return api_result_list

async def multiple_llm_api_request_with_limit (batch_request_list, model_configs, api_limiter):

  llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", **model_configs)

  async def process_single_request(request):
      await api_limiter.wait()
      individual_result = await llm.ainvoke(request)

      result_metadata = individual_result.usage_metadata
      input_tokes = result_metadata.get('input_tokens')
      output_tokes = result_metadata.get('output_tokens')
      result_content = individual_result.content
      return [result_content,input_tokes,output_tokes]
      
  # The for loop append fail because we need a data holder for co-routine to run through not just a data list.
  task_request_list = [process_single_request(request) for request in batch_request_list]
  api_result_list = await(asyncio.gather(*task_request_list))
  return api_result_list

def sorted_multiple_llm_api_result (llm_api_results):
  results_list = []
  input_tokens_list = []
  output_tokens_list = []
  for each_result in llm_api_results:
    #the magic number is the position of result (from: async def process_single_request)
    results_list.append(each_result[0]) 
    input_tokens_list.append(each_result[1])
    output_tokens_list.append(each_result[2])
  data_dictionary = {"results":results_list,"input_tokens":input_tokens_list,"output_tokens":output_tokens_list}
  return data_dictionary

system_prompt = """You are an expert qualitative analyst focusing on Q&A with a question with provided context in the format of Question: (question asked), Context (context provided).
Focus on response to the question do not add any commentary. If the context provided is not relevant simply return: No relevant context found"""

model_configs = {"temperature": 0.0, "max_output_tokens": 120, "thinking_budget": 0}
