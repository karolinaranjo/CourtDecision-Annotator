# Automated Annotation of Court Decisions
## Colombian Constitutional Court

This repository hosts an automated system designed to process and classify decision documents from the Colombian Constitutional Court. It segments texts into specific sections (e.g., antecedentes, consideraciones de la corte, decision, etc.) and identifies text types (regular, quote, emphasis, footnote).

Using some of the functions developed for this endeavor, the user can do the following:


1) **RTF to PDF Conversion**: Streamlines the transformation of RTF documents into PDF format.
2) **PDF to XML Transformation**: Converts PDFs into XML with tags denoting pages, paragraphs, and text blocks (with format information). The system has advanced capabilities to fix some errors related to parser interpretation of blank spaces.
3) **Font Database Creation**: Compiles a database of all fonts (name + size) used in decision document collections.
4) **Document Analysis and Segmentation**: Analyzes a collection of PDF decision documents following the conventions used by the Colombian Constitutional Court, segmenting them into paragraphs and text blocks with assigned sections.
5) **LLM-Based Classification**: Utilizes GPT-4, Llama, and FlanT5 for analyzing and classifying text into sections within judicial decision documents.

## How to Run

To run the full system, it is necessary to accomplish three tasks:

1) Convert PDF Files to Pandas DataFrames
   To process an entire folder, run the function `pdf_files_to_sau` found in `pdf_segmenter_utils.py`:

```python
paragraphs_df, styles_df, documents_df = pdf_files_to_sau(  source_folder,
                                                            target_folder,
                                                            font_db_conn,
                                                            paragraphs_df=None,
                                                            styles_df=None)
```

The function will return three Pandas dataframes ready for further processing.

2) Do the text analysis and assign sections

To assign sections to each row of the relevant dataframes, run the function `update_sections_all_documents(...)` found in `semantic_annotations_utils.py`:

```python
paragraphs_df, styles_df, documents_df=update_sections_all_documents(paragraphs_df, styles_df, documents_df)
```

3) Comparison of the system's annotations with LLM Outputs:

The functions required for this step are found in `llm_benchmarking_utils.py`

To query GPT-4, we use a dataframe structure and send one row at a time for processing. Llama interaction are done via JSON files. Therefore, the system first converts PDF files to JSON and obtains section information from the dataframes in step 2.

```python
out=pdf_files_to_json_for_section_identification_with_llms(pdf_folder, font_db_conn, df_format, df_docs, prompt)
```

- To get GPT-4's section predictions run
  ```python
  df, prompt=json_to_df_for_GPT('json_data_file.json')
  process_excerpts_with_gpt(df, starting_row, ending_row, api_key, prompt)
  ```

- To get Llama's section predictions run:
  
- Few-shot experiments:
   
  GPT-4:
  
  ```python
  shots_for_DF=get_shots_for_DF(df_for_GPT, shots_data, num_shots=5)
  gpt_results_df =process_excerpts_with_gpt_few_shot(df_for_GPT, starting_row, ending_row, api_key, prompt, shots_for_DF)
  ```
  

  Llama:
  ```python
  shots_for_DF=get_shots_for_DF(df_for_GPT, shots_data, num_shots=5)
  shots_for_llama=prepare_json_for_llama_few_shot(df_for_GPT, starting_row, ending_row, prompt, shots_for_DF)
  ```

## Notes
The code also includes functions for balancing data (`balance_data(...)`) and splitting the data into training, validation, and test sets (`split_json_data(...)`)

## Files in the Repository:
- `kutils.py`:  Contains various text processing functions.
- `pdf_segmenter_utils.py`
- `semantic_annotation_utils` : 
- `llm_benchmarking_utils.py`: 
