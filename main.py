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
import streamlit_analytics2 as streamlit_analytics
import re

# For own code process & libraries
import df_processing
import embedding
import llm_api_async

# --------- Setting APIs ---------
#GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key = st.secrets["GOOGLE_API_KEY"])
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
streamlit_analytics.track(unsafe_password=st.secrets["Analytic_password"])

st.set_page_config(layout="wide")
st.logo("Logo.png")
st.title("Analyze qualitative data just like numbers with the Text Calculator.")
st.video("Tutorial_vid.mp4", width=800)
st.divider()
# --------- Collecting User Gmail For futher auth ---------
streamlit_analytics.start_tracking()

user_signup_email = st.text_input("Please enter your gmail to access the software", placeholder="your.email@gmail.com")
email_is_valid = False
# This is key to prevent the loop of session action reset cause declare as variable will get run again
# Using the if statement prevented it!
if "user_logged_in" not in st.session_state: 
    st.session_state["user_logged_in"] = False
    st.session_state["logged_in_email"] = "None"

if user_signup_email != "":
    email_regex = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    if (re.fullmatch(email_regex, user_signup_email)):
        email_is_valid = True
        if st.session_state["user_logged_in"] and st.session_state.get("logged_in_email") == user_signup_email:
            st.success(f"You already logged in for the session as: {user_signup_email}")
        else:
            if st.button("User Session Login In"):
                st.session_state["user_logged_in"] = True
                st.session_state["logged_in_email"] = user_signup_email
                st.success(f"Login as google email: {user_signup_email}")
    else:
        st.write(f"Invalid google email, please try again: {user_signup_email}")

st.divider()

if not st.session_state.get("user_logged_in", False):
    st.error("You need to sign in to access this content.")
    st.stop()

# --------- User Uploading files & previews data ---------
st.subheader("Upload your CSV file here")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columns_list = list(df.columns)

    st.write("File successfully uploaded")
    st.subheader("Data Preview")
    st.write(df.head(3))
    st.markdown("---")
    st.subheader("Analyzing your data: ...")

# --------- User selecting columns for pivot table format ---------
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

# --------- User asking summarizing questions & processing the query ---------

    user_question = st.text_input("Questions to summarize text data:")
    st.write(f"User Query Confirmation: {user_question}")
    run_analysis_button = st.button("Run Analysis")

    if run_analysis_button and user_question != "":
        
        # Dataframe group by
        group_by_table = df_processing.create_groupby_table(df, list(row_vals_list), list(col_vals_list), list(text_field_val_list))
        group_by_table_result = df_processing.pivot_table_result(group_by_table,list(row_vals_list), list(col_vals_list), list(text_field_val_list)).fillna("") # Turn groupbuy count into pivot format
        text_merged_table = df_processing.text_merge_df (df, group_by_table, list(row_vals_list), list(col_vals_list), list(text_field_val_list)) 
        
        # Assume Nan get exclude in groupby_table --> Direct number of APIs calls
        api_current_limit_per_minute = 3900 # 4000 is the actual limit but too tight
        api_limiter = Limiter(api_current_limit_per_minute/60)

        result_tab, result_data_to_download, count_tab, raw_data_tab = st.tabs(["Query Result", "Query Result 4 Download", "Count number of text", "Result & Combined Raw Data"])

# --------- Displaying the results in multiples tabs for download ---------
        with result_tab:
            st.header("User question result")
            new_groupby_table = group_by_table

            estimate_count_down_time = int(round(len(text_merged_table) * (60 / api_current_limit_per_minute) + 4,0))
            df_processing.countdown_clock(estimate_count_down_time)
            new_text_col_list = []
            for each_selected_col in text_field_val_list:
                text_list_per_row = list(text_merged_table[each_selected_col])
                batch_request_list = llm_api_async.batch_prompt_processing(llm_api_async.system_prompt, user_question, context_list=text_list_per_row)
                results = asyncio.run(llm_api_async.multiple_llm_api_request_with_limit(batch_request_list, local_model_configs, api_limiter))
                new_groupby_table[f"{each_selected_col}_result"] = list(results)
                new_text_col_list.append(f"{each_selected_col}_result")
                
            result_pivoted_table = df_processing.pivot_table_result(new_groupby_table, row_vals_list, col_vals_list, new_text_col_list)
            st.table(result_pivoted_table)

        with count_tab:
            st.header("Number of text pivot table")
            st.write(group_by_table_result)

        with raw_data_tab:
            st.header("Query result with merged text context for validation.")
            st.write(text_merged_table)

        with result_data_to_download:
            st.header("Query Result 4 Download")
            st.write(result_pivoted_table)

streamlit_analytics.stop_tracking()

# --------- Contact Information ---------
st.divider()
st.markdown("#### Contact Information")
st.write("If you have feedback, ideas, or just want to share how you used it, I’d really appreciate it.")
st.write("Email: xmb4002@gmail.com")
