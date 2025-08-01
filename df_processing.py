import pandas as pd
import numpy as np
import streamlit as st
import time
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import ast

def create_groupby_table (dataframe, row_vals_list, col_vals_list, text_field_val_list):
  groupby_list = row_vals_list + col_vals_list
  grouped_table = dataframe.groupby(groupby_list)[text_field_val_list].count()

  for select_col in text_field_val_list:
    grouped_table = grouped_table.rename(columns={select_col: f'{select_col}_renamed'})
  # do not reset index because the grouped by columns are the index --> reset --> only return text field val col
  reindex_table = grouped_table.reset_index()

  # This convulted code is needed because if you risk weird behavior (multi variable) text col, that give un expected behavior when reset_index but we also want to keep original cols name
  for select_col in text_field_val_list:
    reindex_table = reindex_table.rename(columns= {f'{select_col}_renamed': select_col})
    reindex_table[select_col] = reindex_table[select_col].astype(str)

  return reindex_table

def text_merge_df (original_df, reindex_table, row_vals_list, col_vals_list, text_field_val_list):
  var_2_filer_list = row_vals_list + col_vals_list

  for index, row in reindex_table.iterrows():
    for select_text_col in text_field_val_list:
      if pd.isna(row[select_text_col]):
        continue
      else:
        temp_df_4_filter = original_df
        # Create & apply filter for each var in var_2_filer_list into a temp_df
        # to get the exact text_column value to correponsding cell of the count() pivot table

        for element in var_2_filer_list:
          current_filter = temp_df_4_filter[element] == row[element]
          temp_df_4_filter = temp_df_4_filter[current_filter]
        text_list = temp_df_4_filter[select_text_col]
        str_text_list = f"{list(text_list)}"
        reindex_table.loc[index, select_text_col] = str_text_list

  return reindex_table

def pivot_table_result (text_merge_df, row_vals_list, col_vals_list, text_field_val_list):
  # aggfunc = first just mean take the first value of the group by which we already calculated
  pivot_table_df = pd.pivot_table(data = text_merge_df, index=row_vals_list,
                                         columns = col_vals_list, values = text_field_val_list, aggfunc='first')
  return pivot_table_df

#print("check dependancy df processing")

def countdown_clock(duration):
    st.write("Data Processing Countdown:")
    countdown_placeholder = st.empty()
    for i in range(duration, -1, -1):
        countdown_placeholder.markdown(f"#### Estimate Time remaining: {i} seconds")
        time.sleep(1)
    countdown_placeholder.write("Collecting Results")

def func_or_filter_list (Combined_filter_list:list) -> list:
  List_combined = np.array(Combined_filter_list).astype(int)
  Index_range_length = len(List_combined[0:1,].tolist()[0])
  filter_list = []
  for each_row_index in range(0,Index_range_length):
    check_or_filter = List_combined[:,each_row_index].sum()
    if check_or_filter > 0:
      filter_list.append(True) 
    else:
      filter_list.append(False)
  return filter_list

def filtering_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  modify_check_box = st.checkbox("Add Filters")
  if modify_check_box == False:
    return df
  
  # To isolate actual df and df use for analysis
  df_copied = df.copy()
  # Try to convert datetimes into standard format (datetime: no timezone)
  for each_column in df_copied.columns:
    if is_object_dtype(df_copied[each_column]):
      try:
        df_copied[each_column] = pd.to_datetime(df_copied[each_column])
      except Exception:
        pass
    
    if is_datetime64_any_dtype(df_copied[each_column]):
      df_copied[each_column] = df[each_column].dt.tz_localize(None)
    
    modification_container = st.container()

    # Create a object container including both selecting to filter & filtering parts
    # Wrong indentation
  with modification_container:
    selected_cols_to_filter = st.multiselect("Filter dataset on:", list(df_copied.columns))
    for each_col_to_filter in selected_cols_to_filter:
      left, right = st.columns((1,20))
      
      if is_categorical_dtype(df_copied[each_col_to_filter]) or df_copied[each_col_to_filter].nunique() <10:
        user_category_input  = right.multiselect(f"Values for: {each_col_to_filter}", 
                                                  df_copied[each_col_to_filter].unique(),
                                                  default=list(df_copied[each_col_to_filter].unique()))
        df_cat_filter = df_copied[each_col_to_filter].isin(user_category_input)
        df_copied = df_copied[df_cat_filter]
      
      elif is_numeric_dtype(df_copied[each_col_to_filter]):
        min_val = float(df_copied[each_col_to_filter].min())
        max_val = float(df_copied[each_col_to_filter].max())
        step = (max_val - min_val) / 100
        user_numerical_input = right.slider(f"Values for: {each_col_to_filter}", min_value = min_val, max_value = max_val, step = step, value=(min_val, max_val))
        df_numerical_filter = df_copied[each_col_to_filter].between(*user_numerical_input) # Fix this: Probably change to manual 2 low high filter.
        df_copied = df_copied[df_numerical_filter]
        st.write(user_numerical_input)

      elif is_datetime64_any_dtype(df_copied[each_col_to_filter]):
        user_date_input = right.date_input(f"Values for {each_col_to_filter}", 
                                            value=(df_copied[each_col_to_filter].min(), df_copied[each_col_to_filter].max()))
        if len(user_date_input) == 2:
          user_date_input = tuple(map(pd.to_datetime, user_date_input))
          start_date, end_date = user_date_input
          df_date_filter = df_copied[each_col_to_filter].between(start_date, end_date)
          df_copied = df_copied[df_date_filter]
      # Next update text filter with and + or methods
      else:
        and_filter_checkbox, or_filter_checkbox = st.columns([1,1])
        with and_filter_checkbox:
          select_and_filter = st.checkbox("Add Filters", key="select_and_filter")
        with or_filter_checkbox:
          select_or_filter = st.checkbox("Or Filters", key="select_or_filter")
        #with semantic_filter_checkbox: # This is an up comming feature
          #sematic_filter = st.checkbox("Semantic Filters", key="select_semantic_filter")

        st.write('Filter substring in the following format: "word1", "word2", "word3" and select either filter by "And" or "Or".')
        user_text_input = right.text_input(f"Substring check {each_col_to_filter}",)
        if user_text_input != "":
          text_list_formated = f"[{user_text_input}]"
          try:
            substring_filter_list = ast.literal_eval(text_list_formated)
            if select_and_filter == True:
              for each_sub_string in substring_filter_list:
                df_text_filter = df_copied[each_col_to_filter].astype(str).str.contains(each_sub_string)
                df_copied = df_copied[df_text_filter]

            elif select_or_filter == True:
              combined_or_filter_list = []
              for each_sub_string in substring_filter_list:
                df_text_filter = df_copied[each_col_to_filter].astype(str).str.contains(each_sub_string)
                combined_or_filter_list.append(list(df_text_filter))
              final_or_filter = func_or_filter_list(combined_or_filter_list)
              df_copied = df_copied[final_or_filter]
              
          except KeyError:
            st.write(text_list_formated)
            st.write(KeyError)
        
  return df_copied

def globalize_localize_column (df: pd.DataFrame) -> pd.DataFrame:
  localize_index = list(range(0,len(df)))
  globalize_index = ["global"] * len(df)
  df["localize"] = localize_index
  df["globalize"] = globalize_index
  return df





