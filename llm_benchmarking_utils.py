import pandas as pd
import json
import pdf_segmenter_utils as kpdf
import kutils as krusty
import os
from os.path import isfile, join
from tqdm import tqdm
import random
import re
import semantic_annotations_utils as sau
from openai import OpenAI


def get_completion_APIV1(client, prompt, model="gpt-4-1106-preview"):
    """
    Connects to an OpenAI client and gets an answer to the prompt using the specified GPT model.

    Parameters
    ----------
    client : OpenAI API client
        Authenticated client object.
    prompt : str
        conversation prompt.
    model : str, optional
        The GPT model to use.

    Returns
    -------
    str
        Answer given by the specified model to the given prompt.
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content



def get_completion_few_shot(client, prompt, input, shots, model="gpt-4-1106-preview"):
    """
    Connects to an OpenAI client and gets an answer to the prompt using the specified GPT model.
    It provides examples from the shots list.

    Parameters
    ----------
    client : OpenAI API client
        Authenticated client object.
    prompt : str
        Initial conversation prompt.
    input : str
        User input for completion
    shots : list of dicts
        Few-shot examples, each containing 'input' and 'output' keys.
    model : str, optional
        The GPT model to use.

    Returns
    -------
    str
        Generated completion for the given prompt, input, and few-shot examples.
    """
    messages = [{"role": "system", "content": prompt}]

    for shot in shots:
        messages.extend([{"role": "user", "content": shot['input']},
                         {"role": "assistant", "content": shot['output']}])

    messages.append({"role": "user", "content": input})

    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content


def get_shots_from_json(json_info, num_shots):
    """
    Retrieves a specified number of shots (examples) randomly from a JSON object.

    Parameters
    ----------
    json_info : list
        List of JSON objects.
    num_shots : int
        Number of shots to randomly select.

    Returns
    -------
    list
        Selected JSON objects.
    """

    indices=random.sample(range(len(json_info)), num_shots)
    shots=[]
    for i in indices:
        shots.append(json_info[i])
    return shots


def shots_for_llama(shots):
    """
    Formats shots (examples) for Llama experiments.

    Parameters
    ----------
    shots : list
       List of shot dictionaries containing 'input' and 'output' keys.

    Returns
    -------
    llama_shots : TYPE
        Formatted shots (examples) for use with Llama.
    """
    
    llama_shots=""
    for shot in shots:
        llama_shots+="\n\n### Input:\n"+shot["input"]+"\n\n### Response:\n"+shot["output"]
    
    return llama_shots

def prepare_json_for_llama_few_shot(df_for_GPT, starting_row, ending_row, prompt, shots_data):
    """
    Prepares JSON data for Llama few-shot learning experiments.

    Parameters
    ----------
    df_for_GPT : pd.DataFrame
        DataFrame containing text data.
    starting_row : int
        Starting row index.
    ending_row : int
        Ending row index.
    prompt : str
        Instruction prompt.
    shots_data : list
        List of shot data.

    Returns
    -------
    list
        JSON output formatted for Llama.
    """

    df=df_for_GPT.reset_index(drop=True)

    df=df.iloc[starting_row:ending_row]
    json_output_for_llama=[]
    for i, row in df.iterrows():
        if len(row['text'])>12:

            shots=shots_data[i]
            
            llama_shots=shots_for_llama(shots)

            json_output_for_llama.append({
                                            "description": "Template used by LLM-Finetuning.",
                                            "response_split": "### Response:",
                                            "instruction":"Following the examples, write a response that appropriately completes the request.\n\n### Instruction:\n"+prompt,
                                            "demo_part":llama_shots,
                                            "query_part":"\n\n### Input:\n"+row["text"]+"\n\n### Response:\n",
                                        })
    return json_output_for_llama


def get_shots_for_DF(df_for_GPT, json_source, num_shots=4):
    """
    Retrieves shots for each row in a DataFrame from a JSON source.

    Parameters
    ----------
    df_for_GPT : pd.DataFrame
        DataFrame containing data for GPT.
    json_source : list
        List of JSON objects to retrieve shots from.
    num_shots : int, optional
        Number of shots to retrieve for each row (default is 4).

    Returns
    -------
    list
        List of shots for each row in the DataFrame.
    """
    shots=[]
    df=df_for_GPT
    for i, row in df.iterrows():
        #get shots
        shots.append(get_shots_from_json(json_source, num_shots))
    return shots

def process_excerpts_with_gpt(df, starting_row, ending_row, api_key, pre_prompt, post_prompt=""):
    """
    Process excerpts from a DataFrame using the OpenAI API.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the excerpts to process.
    starting_row : int
        Starting index of the rows to process.
    ending_row : int
        Ending index of the rows to process.
    api_key : str
        API key.
    pre_prompt : str
        Text to prepend to each excerpt for processing.
    post_prompt : str, optional
        Text to append to each excerpt for processing.

    Returns
    -------
    DataFrame
        DataFrame with processed excerpts and GPT predictions.
    """
    client = OpenAI(api_key=api_key)

    #if column gpt_prediction does not exist, create it and set it to empty string
    if "gpt_prediction" not in df.columns:
        df["gpt_prediction"]=""

    df=df.reset_index(drop=True)
    #get rows from starting_row to ending_row
    df=df.iloc[starting_row:ending_row]
    for i, row in df.iterrows():
        if len(row['text'])>12:
            #get prompt
            prompt=pre_prompt+"\n"+row["text"]+"\n"+post_prompt

            #get completion
            completion=get_completion_APIV1(client, prompt)
            #save completion in dataframe
            df.loc[i, "gpt_prediction"]=completion
            
    return df

def process_excerpts_with_gpt_few_shot(df, starting_row, ending_row, api_key, prompt, shots_data):
    """
    Processes excerpts using OpenAI's GPT model with few-shot learning experiments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the excerpts.
    starting_row : int
        Starting index of the rows to process.
    ending_row : int
        Ending index of the rows to process.
    api_key : str
        API key.
    prompt : str
        Prompt for generating completions.
    shots_data : list
        List of shot data for each excerpt.

    Returns
    -------
    DataFrame
        DataFrame with processed excerpts and GPT predictions for few-shot learning.
    """
    client = OpenAI(api_key=api_key)

    #if column gpt_prediction does not exist, create it and set it to empty string
    if "gpt_prediction" not in df.columns:
        df["gpt_prediction"]=""

    df=df.reset_index(drop=True)

    df=df.iloc[starting_row:ending_row]

    for i, row in df.iterrows():
        if len(row['text'])>12:
            shots=shots_data[i]
            
            completion=get_completion_few_shot(client, prompt, row["text"], shots)
            df.loc[i, "gpt_prediction"]=completion

    return df


def json_to_df_for_GPT(json_file):
    """
    Converts JSON data to a DataFrame for GPT experiments.
    Each JSON object must have fields 'text', 'output', and 'instruction'.

    Parameters
    ----------
    json_file : str
        Path to the JSON file.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with 'text' and 'label' columns.
    prompt: str
        Instruction prompt for GPT.

    """
    with open(json_file) as json_file:
        data= json.load(json_file)
    df = pd.DataFrame(data)
    df.rename(columns = {'input':'text', 'output':'label'}, inplace = True)
    prompt=df.iloc[0]['instruction']
    print(prompt)
    df.drop(['instruction'], axis=1, inplace=True)

    return df, prompt


def json_to_df_for_GPT_v2(json_file):
    """
    Converts JSON data to a DataFrame for GPT experiments.
    Each JSON object must have fields 'text', 'output', and 'prompt'

    Parameters
    ----------
    json_file : str
        Path to the JSON file.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with 'text' and 'label' columns.
    prompt: str
        Instruction prompt for GPT.

    """
    with open(json_file) as json_file:
        data= json.load(json_file)
    df = pd.DataFrame(data)
    df.rename(columns = {'input':'text', 'output':'label'}, inplace = True)
    prompt=df.iloc[0]['prompt']
    print(prompt)
    df.drop(['prompt'], axis=1, inplace=True)

    return df, prompt


def format_df_to_json_merge_paragraphs(df_format, 
                                       target='section_es', 
                                        prompt= "prediga a que sección pertenece el siguiente texto de la sentencia de la corte constitucional. Use la siguiente lista de secciones para la identificación: encabezado, antecedentes, pretensiones, intervenciones, intervención del procurador, norma(s) demandada(s), actuaciones en sede revisión, pruebas, audiencia(s) pública(s), competencia, consideraciones de la corte, síntesis de la decisión, decisión, firmas, salvamento de voto, sin sección. Escoja un elemento de la lista y no diga nada más",
                                       num_paragraphs=3):
    """
    Format metadata DataFrame to JSON by merging paragraphs
    
    Parameters
    ----------
    df_format : pd.DataFrame
        DataFrame containing paragraphs' information.
    target : str, optional
        Column name for the target label. The default is 'section_es'.
    prompt : str, optional
        Instruction prompt for prediction.
    num_paragraphs : int, optional
        Number of paragraphs to merge. The default is 3.

    Returns
    -------
    output : list of dict
        JSON data for Llama experiments.

    """
    output=[]
    curr_count=0
    input_text=""
    for i in tqdm(df_format.index):
        if curr_count<=num_paragraphs and df_format.loc[i, 'document']==df_format.loc[i+1, 'document'] and df_format.loc[i,'section_es']==df_format.loc[i+1,'section_es']:
            curr_count+=1
            input_text+=df_format.loc[i,'text']
        else:
            output.append({'instruction':prompt,
                           "input":input_text+df_format.loc[i,'text'],
                           "output":df_format.loc[i,target]}
                           )
            curr_count=0
            input_text=""
    return output


def visual_tags_for_llms(df_docs, df_semantic, pdfs_path, verbose=False):
    """
     Prepares a Dataframe to be subsequently used to store LLM's results.

    Parameters
    ----------
    df_docs : pd.DataFrame
        DataFrame containing document information.
    df_semantic : DataFrame
        DataFrame containing  excerpts information.
    pdfs_path : str
        Path to the directory containing PDF documents.
    verbose : bool, optional
        Set to true to see detailed processing information.
    Returns
    -------
    df_res: pd.DataFrame
        DataFrame ready to be used to collect LLM prediction results.
        The dataframe has the fields 'document', 'paragraph_id', 'raw_text', 
        'constlaw_es', 'gpt_prediction', 'llama_prediction', 
        'llama_finetuned_prediction'. However,'gpt_prediction', 'llama_prediction', 
        'llama_finetuned_prediction' are all empty.
    """

    font_db_conn=krusty.connect_to_font_db()
    df_semantic=df_semantic.sort_values(by=["document", "paragraph_id", "paragraph_pos"])
    df_semantic=df_semantic.copy()
    df_semantic=df_semantic.dropna()

    df_res=pd.DataFrame(columns=["document", "paragraph_id", "raw_text", "constlaw_es", "gpt_prediction", "llama_prediction", "llama_finetuned_prediction"])

    num_issues=0

    curr_document=-1

    for i, row in df_semantic.iterrows():

        if curr_document!=row["document"]:
            print("----------------------------------------------------------------------") if verbose==True else ""
            print("New Document found") if verbose==True else ""
            #get document that has the id matching the "document" field in the excerpt
            doc=df_docs.loc[df_docs["id"]==row["document"]].iloc[0]
            print("Doc Name:", doc["name"]) if verbose==True else ""
            #get pdf file
            pdf_path=os.path.join(pdfs_path, doc["name"]+".pdf")
            text=kpdf.pdf_to_labeled_text(pdf_path, font_db_conn, font_info=True)
            curr_document=row["document"]
            remaining_text=text

        #find the row text in the text
        pos=remaining_text.find(row["text"])
        if pos==-1:
            print("WARNING: text not found in pdf. Looking for looser matches now ...") if verbose==True else ""
            print("text is", row["text"]) if verbose==True else ""
            num_issues+=1

            search_done=False
            num_words=10
            words=row["text"].split(" ")
            total_words=len(words)
            visited_num_words=[]
            
            while search_done==False:
                
                if num_words==0 or num_words in visited_num_words:
                    print("WARNING: text not found in pdf") if verbose==True else ""
                    search_done=True
                else:
                    visited_num_words.append(num_words)
                    #get the first num_words words of the text
                    shorter_text=" ".join(words[:num_words])
                    print("shorter text:", shorter_text) if verbose==True else ""
                    
                    #try looking for a shorter version of the text
                    ##count the number of matches
                    matches=re.findall(re.escape(shorter_text), remaining_text)
                    num_matches=len(matches)
                    print("number of matches =", num_matches) if verbose==True else ""
                    
                    if num_matches==1:
                        pos=remaining_text.find(shorter_text)
                        search_done=True
                    elif num_matches>1: #make the match more specific by adding more words
                        if num_words==total_words:
                            print("WARNING: There are too many matches.") if verbose==True else ""
                            search_done=True
                        else:
                            num_words+=1 #increase the number of words in shorter_text, this should reduce the number of matches
                    else:#this means that num_matches==0
                        #decrease the number of words in shorter_text
                        num_words-=1

        if pos!=-1: #text was found. It should always be found.
            previous_closing_tag=kpdf.get_last_closing_tag(remaining_text[:pos])
            print("previous closing tag is", previous_closing_tag) if verbose==True else ""
            closing_pos=remaining_text.find(previous_closing_tag)

            if closing_pos==-1:
                print("WARNING: closing tag not found in pdf") if verbose==True else ""
                print("closing tag is", previous_closing_tag) if verbose==True else ""
            else:
                new_start=closing_pos+len(previous_closing_tag)+3
                remaining_text=remaining_text[new_start:]

                res=kpdf.get_tag_instances("font", remaining_text, 1)
                opening_tag=kpdf.get_first_opening_tag2(res)

                _, with_tags, _, _=kpdf.get_text_between_tags3(res, opening_tag)

                df_res.loc[len(df_res)]={"document":row["document"],
                                        "paragraph_id":row["paragraph_id"],
                                        "raw_text":with_tags,
                                        "constlaw_es":row["text_type_es"],
                                        "gpt_prediction":"",
                                        "llama_prediction":"",
                                        "llama_finetuned_prediction":""
                                        }

    return df_res


def convert_df_to_llama_json(df, prompt, json_file="json_output.json", input="raw_text", target="constlaw_es"):
    """
    Converts dataframe to JSON file for Llama experiments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing paragraph data to convert.
    prompt : str
        Instruction prompt for the LLM model.
    json_file : str, optional
        Path to save the JSON file. The default is "json_output.json".
    input : str, optional
        Column name for input text. The default is "raw_text".
    target : str, optional
       Column name for target label. The default is "constlaw_es".

    Returns
    -------
    output : list of dict
        List JSON data ready for Llama prompting.

    """
    output=[]
    for i, row in df.iterrows():
        output.append({"instruction":prompt,
                       "input":row[input],
                       "output":row[target]})

    with open(json_file, 'w') as f:
        json.dump(output, f)

    output=json.dumps(output)
    output=json.loads(output)
    return output

def combine_dataframes(docs1_path, docs2_path, format1_path, format2_path, semantic1_path, semantic2_path):
    """
    Combines data from multiple CSV files into unified dataframes.

    Parameters
    ----------
    docs1_path : str
        Path to the first CSV file containing document data.
    docs2_path : str
        Path to the second CSV file containing document data.
    format1_path : str
        Path to the first CSV file containing paragraph data.
    format2_path : str
        Path to the second CSV file containing paragraph data.
    semantic1_path : str
        Path to the first CSV file containing excerpts data.
    semantic2_path : str
        Path to the second CSV file containing excerpts data.

    Returns
    -------
    df_docs : pd.DataFrame
        Combined DataFrame containing document information.
    df_format : pd.DataFrame
        Combined DataFrame containing paragraph information.
    df_semantic : pd.DataFrame
        Combined DataFrame containing excerpts information.

    """

    # get an unified dataframe with all documents
    df_test1_docs = pd.read_csv(docs1_path)
    df_test2_docs = pd.read_csv(docs2_path)

    df_test1_format = pd.read_csv(format1_path)
    df_test2_format = pd.read_csv(format2_path)

    df_test1_semantic = pd.read_csv(semantic1_path)
    df_test2_semantic = pd.read_csv(semantic2_path)

    # get length of each one of the first dataframes (the ones to add something to)
    len_test1_docs = len(df_test1_docs)
    len_test1_format = len(df_test1_format)
    len_test1_semantic = len(df_test1_semantic)

    # offset df_test2_docs id by len_test1
    df_test2_docs['id'] = df_test2_docs['id'] + len_test1_docs


    # offset df_test2_format "document" by len_test1
    df_test2_format['document'] = df_test2_format['document'] + len_test1_docs
    #offset df_test2_format "id" by len_test1_format
    df_test2_format['id'] = df_test2_format['id'] + len_test1_format

    # offset df_test2_semantic "document" by len_test1
    df_test2_semantic['document'] = df_test2_semantic['document'] + len_test1_docs    
    #offset df_test2_semantic "id" by len_test1_semantic
    df_test2_semantic['id'] = df_test2_semantic['id'] + len_test1_semantic


    # concatenate 
    df_docs = pd.concat([df_test1_docs, df_test2_docs], ignore_index=True)
    df_format = pd.concat([df_test1_format, df_test2_format], ignore_index=True)
    df_semantic = pd.concat([df_test1_semantic, df_test2_semantic], ignore_index=True)

    df_docs = df_docs[['id', 'name', 'type', 'year', 'num_sections', 'num_paragraphs']]
    df_format=df_format[['id', 'document', 'paragraph_id', 'previous', 'next','length', 'document_pos', 'section', 'function', 'text']]
    df_semantic=df_semantic[['id', 'document', 'paragraph_id', 'text_type', 'length', 'document_pos', 'paragraph_pos', 'function', 'text', 'section']]

    return df_docs, df_format, df_semantic

def df_to_json(df,target='text_type_es', prompt=""):
    """
    Converts DataFrame (columns 'text' and target) to JSON format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data.
    target : str, optional
        Column name for target label. The default is 'text_type_es'.
    prompt : str
        Instruction prompt for JSON.
        
    Returns
    -------
    output : list of dict
        List of dictionaries representing JSON data.

    """
    if prompt=="":
        print("WARNING: prompt is empty")
    else:
        init_prompt=prompt
        output=[]
        for i in tqdm(df.index):
            output.append({'instruction':init_prompt,
                        "input":df.loc[i,'text'],
                        "output":df.loc[i,target]}
                        )
        return output
    
def get_paragraphs_sections(paragraphs, df_paragraphs, df_docs, doc_name):
    """
    Assigns a document's paragraphs to the corresponding sections within the document.

    Parameters
    ----------
    paragraphs : list of str
        List of paragraphs of the document.
    df_paragraphs : pd.DataFrame
        DataFrame containing paragraph information for the collection of documents.
    df_docs : DataFrame
        DataFrame containing document information.
    doc_name : str
        Name of the document.
    Returns
    -------
    sections : list of str
        List of sections corresponding to each paragraph.

    """
    sections=[]
    for paragraph in paragraphs:

        _, tags = kpdf.get_initial_tags(paragraph)

        paragraph=kpdf.remove_tags(paragraph, tags).strip()
        #get document id
        id=sau.get_document_id(doc_name, df_docs)

        matches=df_paragraphs.loc[(df_paragraphs["document"]==id)]
        matches=matches.copy()
        matches["text"]=matches["text"].apply(lambda x: re.escape(x))
        matches=matches.loc[(matches["text"].str.contains(re.escape(paragraph)))]

        if len(matches)==1:
            #get section
            sections.append(matches.iloc[0]["section_es"])
        else:
            sections.append("unmatched")

    return sections


def prepare_document_json_for_section_identification(doc_xml, doc_name, df_format, df_docs, prompt, num_paragraphs_per_query, overlap):
    """
    Prepares JSON data for section classification using Llama.

    Parameters
    ----------
    doc_xml : str
        decision document XML-file.
    doc_name : str
        Name of the document.
    df_format : pd.DataFrame
        DataFrame containing formatted paragraph information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str
        Instruction prompt
    num_paragraphs_per_query : int
        Number of paragraphs per query.
    overlap : int
        Number of overlapping paragraphs between queries.

    Returns
    -------
    json : list of dict
        JSON data prepared for section classification. Paragraphs are wrapped in <paragraph>...</paragraph> tags.

    """
    paragraphs=kpdf.get_paragraphs2(doc_xml)

    sections=get_paragraphs_sections(paragraphs, df_format, df_docs, doc_name)

    print("sections are", set(sections))
    json=[]
    i=0
    while i < len(paragraphs):
        text=""
        curr_section=sections[i]
        section="Sin sección"
        for j in range(i, i+num_paragraphs_per_query):

            if j<len(paragraphs):
                text+="<paragraph>"+paragraphs[j]+"</paragraph>"
                if curr_section!=sections[j] and sections[j]!="unmatched":
                    section=sections[j]

        i=i+num_paragraphs_per_query-overlap
        json.append({"instruction":prompt,
                     "input":text,
                     "output":section})

    return json

def prepare_document_json_for_section_identification2(doc_xml, doc_name, df_format, df_docs, prompt, num_paragraphs_per_query, overlap):
    """
    Prepares JSON data for section classification using Llama.

    Parameters
    ----------
    doc_xml : str
        decision document XML-file.
    doc_name : str
        Name of the document.
    df_format : pd.DataFrame
        DataFrame containing formatted paragraph information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str
        Instruction prompt
    num_paragraphs_per_query : int
        Number of paragraphs per query.
    overlap : int
        Number of overlapping paragraphs between queries.

    Returns
    -------
    json : list of dict
        JSON data prepared for section classification.

    """

    paragraphs=kpdf.get_paragraphs2(doc_xml)

    sections=get_paragraphs_sections(paragraphs, df_format, df_docs, doc_name)

    paragraphs=kpdf.remove_tags_from_paragraphs(paragraphs)

    print("sections are", set(sections))
    json=[]
    i=0
    while i < len(paragraphs):
        text=""

        curr_section=sections[i]

        section="Sin sección"
        for j in range(i, i+num_paragraphs_per_query):

            if j<len(paragraphs):

                text+=paragraphs[j]
                if curr_section!=sections[j] and sections[j]!="unmatched":
                    section=sections[j]


        i=i+num_paragraphs_per_query-overlap
        json.append({"instruction":prompt,
                     "input":text,
                     "output":section})

    return json


def pdf_files_to_json_for_section_identification_with_llms(input_folder, font_db_conn, paragraphs_df, df_docs, prompt="", num_paragraphs_per_query=5, overlap=2):
    """
    Convert PDF files to JSON for section classification with LLMs (Llama syntax).

    Parameters
    ----------
    input_folder : str
        Path to the folder containing PDF files.
    font_db_conn : sqlite3.Connection
        Connection to the font database.
    df_format : pd.DataFrame
        DataFrame containing formatted paragraph information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str
        Instruction prompt
    num_paragraphs_per_query : int, optional
        Number of paragraphs per query. The default is 5.
    overlap : nt, optional
        Number of overlapping paragraphs between queries. The default is 2.

    Returns
    -------
    json : list of dict
        JSON data prepared for section classification. Paragraphs are wrapped in <paragraph>...</paragraph> tags.

    """
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")    
    files=os.listdir(py_input_folder)

    json=[]

    for f in files:

        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):
            
            xml=kpdf.pdf_to_labeled_text(file_path, font_db_conn, font_info=True)
            
            print("f is", f)

            doc_json=prepare_document_json_for_section_identification(xml, f, paragraphs_df, df_docs, prompt, num_paragraphs_per_query, overlap)

            json+=doc_json
            
    return json

def pdf_files_to_json_for_section_identification_with_llms2(input_folder, font_db_conn, paragraphs_df, df_docs, prompt="", num_paragraphs_per_query=5, overlap=2):
    """
    Convert PDF files to JSON for section classification with LLMs (Llama syntax).

    Parameters
    ----------
    input_folder : str
        Path to the folder containing PDF files.
    font_db_conn : sqlite3.Connection
        Connection to the font database.
    df_format : pd.DataFrame
        DataFrame containing formatted paragraph information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str
        Instruction prompt
    num_paragraphs_per_query : int, optional
        Number of paragraphs per query. The default is 5.
    overlap : nt, optional
        Number of overlapping paragraphs between queries. The default is 2.

    Returns
    -------
    json : list of dict
        JSON data prepared for section classification.

    """
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")    
    files=os.listdir(py_input_folder)

    json=[]

    for f in files:

        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):
            
            xml=kpdf.pdf_to_labeled_text(file_path, font_db_conn, font_info=True)
            
            print("f is", f)

            doc_json=prepare_document_json_for_section_identification2(xml, f, paragraphs_df, df_docs, prompt, num_paragraphs_per_query, overlap)

            #join the two lists
            json+=doc_json
            
    return json

def prepare_document_json_for_style_identification(doc_xml, doc_name, df_styles, df_docs, prompt):
    """
    Prepares JSON data for excerpt type classification using Llama.

    Parameters
    ----------
    doc_xml : str
        decision document XML-file.
    doc_name : str
        Name of the document.
    df_styles : pd.DataFrame
        Dataframe containing excerpts information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str
        Instruction prompt

    Returns
    -------
    json : list of dict
        JSON data prepared for excerpt type classification.

    """
    id=sau.get_document_id(doc_name, df_docs)

    paragraphs=kpdf.get_paragraphs2(doc_xml)

    json=[]

    for paragraph in tqdm(paragraphs):

        remaining_text=paragraph.replace("<paragraph>\n <f", "<paragraph><f")
        

        while remaining_text!="":
            #get first tag
            opening_tag=kpdf.get_first_opening_tag2(remaining_text)
            if opening_tag!="d'oh":

                #get text between tags
                text_without_tags, text_with_tags, _, _=kpdf.get_text_between_tags3(remaining_text, opening_tag)
                matches=df_styles.loc[(df_styles["document"]==id) & (df_styles["text"]==text_without_tags)]
                if len(matches)==1:
                    style=matches.iloc[0]["text_type_es"]

                    json.append({"instruction":prompt,
                                "input":text_with_tags,
                                "output":style})
                    
                #trim text to remove text with tags
                remaining_text=remaining_text.replace(text_with_tags, "")

            else:
                remaining_text=""

    return json

def pdf_files_to_json_for_style_identification_with_llms(input_folder, font_db_conn, df_styles, df_docs, prompt="",):
    """
    Convert PDF files to JSON for excerpts classification with LLMs (Llama syntax).

    Parameters
    ----------
    input_folder : str
        Path to the folder containing PDF files.
    font_db_conn : sqlite3.Connection
        Connection to the font database.
    df_styles : pd.DataFrame
        Dataframe containing excerpts information.
    df_docs : pd.DataFrame
        DataFrame containing document information.
    prompt : str, optional
        Instruction prompt.

    Returns
    -------
    json : list of dict
        JSON data prepared for excerpt classification.

    """
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")    
    files=os.listdir(py_input_folder)

    json=[]

    for f in tqdm(files):

        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):

            xml=kpdf.pdf_to_labeled_text(file_path, font_db_conn, font_info=True)
            
            print("f is", f)

            doc_json=prepare_document_json_for_style_identification(xml, f, df_styles, df_docs, prompt)

            json+=doc_json
            
    return json

def split_json_data(json_data, train_size=0.7, val_size=0.2):
    """
    Split JSON data into training, validation, and test sets.

    Parameters
    ----------
    json_data : list of dict
        JSON data to split.
    train_size : float, optional
        Size of the training set as a fraction of the total data. Default is 0.7.
    val_size : float, optional
        Size of the validation set as a fraction of the total data. Default is 0.2.
        
    Returns
    -------
    train_data : list of dict
        Training set.
    val_data : list of dict
        Validation set.
    test_data : list of dict
        Test set.
    """
    random.shuffle(json_data)

    total_size=len(json_data)
    train_size=int(total_size*train_size)
    val_size=int(total_size*val_size)

    train_data=json_data[:train_size]
    val_data=json_data[train_size:train_size+val_size]
    test_data=json_data[train_size+val_size:]
    
    return train_data, val_data, test_data

def balance_data(set_to_balance, field_to_balance="output", majority_value="Sin sección", proportion_to_keep=0.05):
    """
    Balance dataset by randomly sampling a proportion of the majority class and removing the rest.

    Parameters
    ----------
    set_to_balance : list of dict
        Dataset to balance.
    field_to_balance : str, optional
        Field name for balancing. Default: "output".
    majority_value : str, optional
        Majority class value. Default: "Sin sección".
    proportion_to_keep : float, optional
        Proportion of majority class to retain. Default: 0.05.

    Returns
    -------
    balanced_set : list of dict
        Balanced dataset.

    """
    indices_majority=[i for i, x in enumerate(set_to_balance) if x[field_to_balance] == majority_value]
    indices_minority=[i for i, x in enumerate(set_to_balance) if x[field_to_balance] != majority_value]

    proportion_to_keep=0.05
    random.seed(42)
    
    indices_majority_to_keep=random.sample(indices_majority, int(len(indices_majority)*proportion_to_keep))
    #len(indices_sin_seccion_to_keep)

    balanced_set=[i for j, i in enumerate(set_to_balance) if j in indices_majority_to_keep or j in indices_minority]

    random.shuffle(balanced_set)
    len(balanced_set)
    return balanced_set


def strip_json_from_tags(json_data):
    """
    Remove tags from input text in a list of JSON data.

    Parameters
    ----------
    json_data : list of dict
        List of JSON data.

    Returns
    -------
    json_data : list of dict
        List of JSON data with tags removed from input text.
    """
    
    for i in range(len(json_data)):
        aux=json_data[i]['input']
        _, tags = kpdf.get_initial_tags(aux)
        new_text=kpdf.remove_tags(aux, tags) 
        json_data[i]['input']=new_text
    return json_data

def modify_json_prompt(json_data, prompt):
    """
    Modify the instruction prompt in a list of JSON data.

    Parameters
    ----------
    json_data : list of dict
        List of JSON data.
    prompt : str
        The new instruction prompt.

    Returns
    -------
    json_data : list of dict
        List of JSON data with updated instruction prompt.
    """

    for i in range(len(json_data)):
        json_data[i]['instruction']=prompt

    return json_data


def extract_class(text_list,class_list):
    """
    Extracts the first class found in a list of classes from a list of text.

    Parameters
    ----------
    text_list : list
        List of text.
    class_list : list
        List of classes.

    Returns
    -------
    str
        The first class found in the text list, or "no class" if no class is found.

    """
  for c in class_list:
    if c in text_list:
      return c
  return "no class"


def compute_accuracy(df, label, prediction):

    accuracy= df[df[label]==df[prediction]].shape[0]/df.shape[0]
    print('accuracy =', accuracy)
    return accuracy
