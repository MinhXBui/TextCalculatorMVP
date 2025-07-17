from google import genai
from google.genai import types
import os

def embedding_selected_cols (dataframe, text_cols_list):
  for select_col in text_cols_list:
    text_list_2_embed = list(dataframe[select_col])
    result = client.models.embed_content(model="gemini-embedding-001", contents = text_list_2_embed,
            config=types.EmbedContentConfig(output_dimensionality=768, task_type="QUESTION_ANSWERING"))
    embedded_result_list = []
    for embed_content in result.embeddings:
      embedded_result_list.append(embed_content.values)
    dataframe[f'{select_col}_embed'] = embedded_result_list
  return dataframe

def question_embedding (user_question:str):
  result = client.models.embed_content(model="gemini-embedding-001", contents = user_question,
            config=types.EmbedContentConfig(output_dimensionality=768, task_type="QUESTION_ANSWERING"))
  user_question_embeded = result.embeddings[0].values
  return user_question_embeded
