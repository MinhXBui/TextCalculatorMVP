import pandas as pd
import streamlit as st
import time

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
    st.write("Result Countdown:")
    countdown_placeholder = st.empty()
    for i in range(duration, -1, -1):
        countdown_placeholder.markdown(f"#### Time remaining: {i} seconds")
        time.sleep(1)
    countdown_placeholder.write("Result collected! 🎉")