#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:50:40 2023

@author: KarolinaNaranjo
"""

import openai
#from openai.api_resources import model
import pandas as pd
import json
import pdf_segmenter_utils as kpdf
import kutils as krusty
import os
from tqdm import tqdm
import random
import re

openai.api_key = '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'

def get_completion(prompt, model="gpt-4-1106-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def process_excerpts_with_gpt(df, starting_row, ending_row, pre_prompt="", post_prompt=""):
    
    #if column gpt_prediction does not exist, create it and set it to empty string
    if "gpt_prediction" not in df.columns:
        df["gpt_prediction"]=""


    if pre_prompt=="":
        pre_prompt=f""" prediga a que sección pertenece el siguiente texto de la sentencia de la corte constitucional \

        Use la siguiente lista de secciones para la identificación:\
        encabezado
        antecedentes
        actuaciones en sede revisión
        intervenciones
        intervención del procurador
        pruebas
        audiencia pública
        competencia
        consideraciones de la corte
        síntesis de la decisión
        decisión
        salvamento de voto
        sin sección
        norma(s) demandada(s)

        Escoja un elemento de la lista y no diga nada más.
        """

    #reset index
    df=df.reset_index(drop=True)
    #get rows from starting_row to ending_row
    df=df.iloc[starting_row:ending_row]
    for i, row in df.iterrows():
        if len(row['text'])>12:
            #get prompt
            prompt=pre_prompt+"\n"+row["text"]+"\n"+post_prompt

            #get completion
            completion=get_completion(prompt)
            #save completion in dataframe
            df.loc[i, "gpt_prediction"]=completion
    return df


def json_to_df_for_GPT(json_file):
    with open(json_file) as json_file:
        data= json.load(json_file)
    df = pd.DataFrame(data)
    df.rename(columns = {'input':'text', 'output':'label'}, inplace = True)
    prompt=df.iloc[0]['instruction']
    print(prompt)
    df.drop(['instruction'], axis=1, inplace=True)

    #drop rows in which text is less than 12 characters
    # df=df[df['text'].str.len()>12]
    # df=df.reset_index(drop=True)

    return df, prompt

def extract_class(text_list,class_list):
  for c in class_list:
    if c in text_list:
      return c
  return "no class"



def format_df_to_json_merge_paragraphs(df_format, 
                                       target='section_es', 
                                        prompt= "prediga a que sección pertenece el siguiente texto de la sentencia de la corte constitucional. Use la siguiente lista de secciones para la identificación: encabezado, antecedentes, pretensiones, intervenciones, intervención del procurador, norma(s) demandada(s), actuaciones en sede revisión, pruebas, audiencia(s) pública(s), competencia, consideraciones de la corte, síntesis de la decisión, decisión, firmas, salvamento de voto, sin sección. Escoja un elemento de la lista y no diga nada más",
                                       num_paragraphs=3):

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


def semantic_df_for_GPT(df_docs, df_format, df_semantic, pdfs_path, num_samples=1, verbose=False):

    font_db_conn=krusty.connect_to_font_db()

    excerpts=df_format.sample(n=num_samples)

    df_semantic=df_semantic.copy()
    
    df_semantic=df_semantic.dropna()

    df_res=pd.DataFrame(columns=["document", "paragraph_id", "raw_text", "constlaw_es", "gpt_prediction", "llama_prediction", "llama_finetuned_prediction"])

    for i, excerpt in excerpts.iterrows():
        #get document that has the id matching the "document" field in the excerpt
        doc=df_docs.loc[df_docs["id"]==excerpt["document"]].iloc[0]
        #get pdf file
        pdf_path=os.path.join(pdfs_path, doc["name"]+".pdf")
        text=kpdf.pdf_to_labeled_text(pdf_path, font_db_conn, font_info=True)

        res=kpdf.get_tag_instances("font", text, 1)
        opening_tag=kpdf.get_first_opening_tag2(res)

        semantic_related=df_semantic.loc[df_semantic["document"]==excerpt["document"]]

        text_without_tags, _, _, _=kpdf.get_text_between_tags3(res, opening_tag)

        #get row in which text contains the text without tags
        row=semantic_related.loc[semantic_related["text"] == text_without_tags]

        #if there is more than one row, send a warning
        if len(row)>1:
            print("WARNING: there is more than one row with the same text") if verbose==True else ""
            aux_row=row.loc[row["text"]==text_without_tags]
            if len(aux_row)>0:
                row=aux_row
            print(row) if verbose==True else ""
        elif len(row)==0:
            print("WARNING: there is no row with the same text") if verbose==True else ""
            
        else:
            print(row) if verbose==True else ""
        #add entry to dataframe

        if len(row)>0:

            df_res.loc[len(df_res)]={"document":excerpt["document"],
                                    "paragraph_id":row["paragraph_id"].iloc[0],
                                    "raw_text":res,
                                    "constlaw_es":row["text_type_es"].iloc[0],
                                    "gpt_prediction":"",
                                    "llama_prediction":"",
                                    "llama_finetuned_prediction":""
                                    }

        print(res) if verbose==True else ""

    return df_res


def semantic_df_for_GPT2(df_docs, df_format, df_semantic, pdfs_path, verbose=False):

    font_db_conn=krusty.connect_to_font_db()

    #excerpts is df_semantic sorted by document, paragraph_id, and id
    excerpts=df_semantic.sort_values(by=["document", "paragraph_id", "id"])

    print("number of excerpts is", len(excerpts))

    #create new dataframe
    df_res=pd.DataFrame(columns=["document", "paragraph_id", "raw_text", "constlaw_es", "gpt_prediction", "llama_prediction", "llama_finetuned_prediction"])

    current_doc=-1
    current_start=0

    for i, excerpt in excerpts.iterrows():
        if current_doc!=excerpt["document"]:
            #get document that has the id matching the "document" field in the excerpt
            doc=df_docs.loc[df_docs["id"]==excerpt["document"]].iloc[0]
            #get pdf file
            pdf_path=os.path.join(pdfs_path, doc["name"]+".pdf")
            original_text=kpdf.pdf_to_labeled_text(pdf_path, font_db_conn, font_info=True)
            current_doc=excerpt["document"]
            current_start=0
        else:
            if pos!=-1:
                current_start+=pos

        text=original_text[current_start:]


        # print("----------------------------")
        # print("current text is", text)

        #pos=text.find(excerpt["text"])
        matches=re.search(excerpt["text"], text)

        print("matches are", matches)
        if len(matches)>0:
            pos=text.find(matches[0])
        else:
            pos=-1

        # print("pos is", pos)
        if pos==-1:
            print("WARNING: text not found in pdf")
            print("text is", excerpt["text"])
        else:
            res=kpdf.get_tag_instances("font", text, 1)
            #opening_tag=kpdf.get_first_opening_tag2(res)

            #add entry to dataframe
            df_res.loc[len(df_res)]={"document":excerpt["document"],
                                    "paragraph_id":excerpt["paragraph_id"],
                                    "raw_text":res,
                                    "constlaw_es":excerpt["text_type_es"],
                                    "gpt_prediction":"",
                                    "llama_prediction":"",
                                    "llama_finetuned_prediction":""
                                    }

            # print(res)

    return df_res


def convert_df_to_llama_json(df, prompt, json_file="json_output.json", input="raw_text", target="constlaw_es"):
    output=[]
    for i, row in df.iterrows():
        output.append({"instruction":prompt,
                       "input":row[input],
                       "output":row[target]})
    #convert to json
    with open(json_file, 'w') as f:
        json.dump(output, f)

    output=json.dumps(output)
    output=json.loads(output)


    return output

def combine_dataframes(docs1_path, docs2_path, format1_path, format2_path, semantic1_path, semantic2_path):

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

def df_to_json(df,target='text_type_es', prompt= prompt):
    init_prompt=prompt
    output=[]
    for i in tqdm(df.index):
        output.append({'instruction':init_prompt,
                       "input":df.loc[i,'text'],
                       "output":df.loc[i,target]}
                       )
    return output