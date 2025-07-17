# General purpose & processing libraries
import numpy as np
import time
import json
import tqdm
from tqdm import tqdm
import pandas as pd

# For Emdedding & LLM async data summerization 
import os
import asyncio
from asynciolimiter import Limiter
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity

# For testing streamlit
import subprocess
import streamlit as st

# For own code process & libraries
import df_processing
import embedding
import llm_api_async

# ---- Setting APIs ----
#GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key = st.secrets["GOOGLE_API_KEY"])
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(layout="wide")
st.title("""Summarize text like numbers with ⛏️ 10ft Deep Insights """)
st.divider()
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columns_list = list(df.columns)

    st.write("File successfully uploaded")
    st.subheader("Data Preview")
    st.write(df.head(3))
    st.markdown("---")
    st.subheader("Analyzing your data: ...")

    data_filter_section, process_setting_section = st.columns([1,1])
    with data_filter_section:
        st.write("Settings for pivot table: (Do not select duplicated cols)")
        row_vals_list = st.multiselect("Select pivot table rows:", columns_list)
        col_vals_list = st.multiselect("Select pivot table columns:", columns_list)

    with process_setting_section:
        st.write("Additional output adjustment:")
        brevity_setting_list = ["Very brief", "Concise", "Standard", "Detailed", "Extensive"]
        #output_type = ["Simple text merge", "Sentiment Anlysis", "LLM Output"]
        text_field_val_list = st.multiselect("Select text columns to analyze:", columns_list)
        llm_brevity_setting = st.selectbox("Select the results length:", brevity_setting_list)
        token_output_dict = {"Very brief":15,"Concise":50,"Standard":140,"Detailed":350,"Extensive":700}
        local_model_configs = llm_api_async.model_configs
        local_model_configs["max_output_tokens"] = token_output_dict[llm_brevity_setting]

    user_question = st.text_input("Questions to summarize text data:")
    st.write(f"User Query Confirmation: {user_question}")
    run_analysis_button = st.button("Run Analysis")

    if run_analysis_button and user_question != "":
        
        # Dataframe group by
        group_by_table = df_processing.create_groupby_table(df, list(row_vals_list), list(col_vals_list), list(text_field_val_list))
        group_by_table_result = df_processing.pivot_table_result(group_by_table,list(row_vals_list), list(col_vals_list), list(text_field_val_list)).fillna("")
        text_merged_table = df_processing.text_merge_df (df, group_by_table, list(row_vals_list), list(col_vals_list), list(text_field_val_list))

        result_tab, result_data_to_download, count_tab, raw_data_tab = st.tabs(["Query Result", "Query Result 4 Download", "Count number of text", "Text merged data"])
        with result_tab:
            st.header("User question result")
            new_groupby_table = group_by_table
            df_processing.countdown_clock(3)
            new_text_col_list = []
            for each_selected_col in text_field_val_list:
                text_list_per_row = list(text_merged_table[each_selected_col])
                batch_request_list = llm_api_async.batch_prompt_processing(llm_api_async.system_prompt, user_question, context_list=text_list_per_row)
                results = asyncio.run(llm_api_async.multiple_llm_api_request(batch_request_list, local_model_configs))
                new_groupby_table[f"{each_selected_col}_result"] = list(results)
                new_text_col_list.append(f"{each_selected_col}_result")
                
            result_pivoted_table = df_processing.pivot_table_result(new_groupby_table, row_vals_list, col_vals_list, new_text_col_list)
            st.table(result_pivoted_table)

        with count_tab:
            st.header("Number of text pivot table")
            st.write(group_by_table_result)

        with raw_data_tab:
            st.header("Text merged unpivot data table for download")
            st.write(text_merged_table)

        with result_data_to_download:
            st.header("Query Result 4 Download")
            st.write(result_pivoted_table)

st.divider()
st.write(" ")
st.title("Contact information & social media.") 
st.write("Email: xmb4002@gmail.com")
st.write("X/Twitter: @TextCalculator")