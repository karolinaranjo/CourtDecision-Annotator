import pdf_segmenter_utils as kpdf
import kutils as krusty
import pandas as pd
import llm_benchmarking_utils as llm_utils
import json
import time


df_docs = pd.read_csv('annotated_data/df_docs.csv')
df_format = pd.read_csv('annotated_data/df_format_es.csv')
df_semantic = pd.read_csv('annotated_data/df_semantic_es.csv')

start_time = time.time()

df_res = llm_utils.semantic_df_for_GPT(df_docs, df_format, df_semantic, "PDFs", num_samples=30000)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")

df_res.to_csv('annotated_data/df_res_semantic.csv', index=False)