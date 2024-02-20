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