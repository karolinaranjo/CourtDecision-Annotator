#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:11:57 2023
Updated on Thu Oct 19 2023

@author: KarolinaNaranjo
"""

import numpy as np

import pdf_segmenter_utils as kpdf
import pandas as pd
import os
from os.path import isfile, join
import kutils as krusty
import re

import math

import spacy

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from IPython.display import display

import Levenshtein


def get_text_type(tag, tagless_text, paragraph_text, usual_font_size):
    '''
    Takes in a tag, tagless text, paragraph text, and usual font size as input 
    and returns the text type based on the font size and other attributes.

    Parameters
    ----------
    tag : Str
        The tag associated with the text
    tagless_text : Str
        text without any tags.
    paragraph_text : Str
        text of the paragraph.
    usual_font_size : Int
        The usual font size of the text.

    Returns
    -------
    text_type : Str
        The type of the text based on its attributes:'title', 'footnote', 
        'emphasis', 'quote', and 'plain'.

    '''
    #get font size
    fs=kpdf.get_trailing_number(tag)
    #print("Font size is ", fs)
    # print("Usual font size is ", usual_font_size)
    
    text_type=""
    #use font size to establish type
    if fs>usual_font_size:
        text_type=add_info_to_string(text_type,"title")
        
    elif fs<usual_font_size:
        text_type=add_info_to_string(text_type,"footnote")

    # if tagless_text.isupper() and "title" not in text_type:
    #     text_type=add_info_to_string(text_type,"title")
        
    
    if tagless_text.isupper() and "title" not in text_type:
        text_type=add_info_to_string(text_type,"emphasis") 

    if "italic" in tag:
        text_type=add_info_to_string(text_type,"quote")
    
    if "bold" in tag and "emphasis" not in text_type:
        text_type=add_info_to_string(text_type,"emphasis")
   
    if text_type=="":
        text_type="plain"

    return text_type


def add_info_to_string(old_str, param):
    '''
    Accepts an existing string and a parameter as inputs, and produces a new string 
    that is a combination of the existing string and the parameter, with an underscore separating them.

    Parameters
    ----------
    old_str : Str
        Old string.
    param : Str
        parameter to be added to the old string.

    Returns
    -------
    new_str : Str
        Updated string separated by an underscore.

    '''
    if len(old_str)==0:
        new_str=param
    else:
        new_str=old_str+"_"+param
    return new_str

def process_paragraph_text(text, doc_id, paragraph_id, paragraph_text, df, usual_font_size, paragraph_doc_pos=0):
    '''
    Processes the text paragraph and returns a dataframe with information about the text (XML representation of the paragraph)
    List of tuples containing the text type and text length.

    Parameters
    ----------
    text : Str
        Paragraph text with inner tags (if any).
    doc_id : Int
        Document ID.
    paragraph_id : int
        Paragraph identifier.
    paragraph_text : Str
        Should be raw text (no tags).
    df : Pandas dataframe 
        It contains OHCO for a document collection.
    usual_font_size : int
        The usual font size of the text.
    paragraph_doc_pos : int, optional
        The position of the paragraph in the document. The default is 0.

    Returns
    -------
    df : Pandas dataframe
       updated dataframe with information about the text.
    par_xml : str
        An XML representation of the paragraph.
    par_info : List[Tuple[str, int]]
        A list of tuples containing the text type and text length.

    '''

    new_text=text
    #print("paragraph: ", new_text)
    first_tag=kpdf.get_first_opening_tag(new_text)
    
    par_xml=""
    par_info=[]
    cursor=0

    if first_tag!="d'oh": #if there are tags to be processed
        
        end_of_paragraph=False
        while not end_of_paragraph:
            
            #print("first_tag is "+ first_tag)
            
            tagless_text, tagged_text, ini_tagless, ini_tagged=kpdf.get_text_between_tags2(new_text, first_tag)
            
            text_length=len(tagless_text)
            cursor=cursor+ini_tagged+len(tagged_text)
            
            #cursor=new_text.find()

            #print("tagless text: ", "'"+tagless_text+"'")
            #print("length tagless text: ", text_length)
        
            #print("tagged_text: ", "'"+tagged_text+"'")
            #print("length tagged_text: ", len(tagged_text))

            #get text type
            text_type = get_text_type(first_tag, tagless_text, paragraph_text, usual_font_size)
            
            #add to list of text types
            par_info.append((text_type, text_length))
            
            paragraph_pos= paragraph_text.find(tagless_text)
            doc_pos=paragraph_doc_pos + paragraph_pos
            
            #save to dataframe
            df.loc[len(df)] = [doc_id, paragraph_id, "Pending", text_type, text_length, doc_pos, paragraph_pos, "Pending", tagless_text]

            #save to xml
            #print("text_type", text_type)
            par_xml+="<"+text_type+">"+tagless_text+"</"+text_type+">"
            
            #print(par_xml)
            #print("cursor:", cursor)
            #print("text is "+text)
            new_text=text[cursor:]
            #print("new text is "+new_text)
            first_tag=kpdf.get_first_opening_tag(new_text)
            if first_tag=="d'oh": #no inner tag, we're done
                end_of_paragraph=True
                #print("end of current paragraph")
    
    return df, par_xml, par_info

def get_semantic_blocks_from_XML(full_text, doc_id, paragraphs_df=None, styles_df=None):
    """
    Processes a document’s text and ID, and updates dataframes for paragraphs and styles. 

    Parameters
    ----------
    full_text : Str
        The full text of the document.
    doc_id : Int
        The ID of the document.
    paragraphs_df : Pandas dataframe, optional
        A DF containing information about the paragraphs in the document.
        The default is None.
    styles_df : Pandas dataframe optional
        A dataframe containing information about the styles in the document. 
        The default is None.

    Returns
    -------
    styles_df : pd.DataFrame
        The updated dataframe containing information about the styles
    paragraphs_df : pd.DataFrame
        The updated dataframe containing information about the paragraphs.
    sections : List[Tuple[str, str]]
        A list of tuples containing the section name and its corresponding text.
    doc_xml : str
        An XML representation of the document.

    """

    _,tags=kpdf.get_tags(full_text)
    usual_font_size=kpdf.get_most_common_font_size(tags)
    print("Document font size: ", usual_font_size)
    paragraphs=kpdf.get_paragraphs(full_text)

    #check if df is empty
    if styles_df is None:
        print("Creating new styles dataframe...")
        
        styles_df=pd.DataFrame(columns=['document', 'paragraph_id', 'section', 'text_type', 'length','document_pos', 'paragraph_pos', 'function','text'])
        #name index column
        styles_df.index.name = 'id'
        
    #check if df is empty
    if paragraphs_df is None:
        print("Creating new paragraphs dataframe...")
        
        paragraphs_df=pd.DataFrame(columns=['document', 'paragraph_id', 'previous', 'next','length','document_pos', 'section', 'function', 'text'])
        #name index column
        paragraphs_df.index.name = 'id'
    
    doc_xml="<document>\n"
    current_section="header"
    sections=[]
    doc_pos=0
    

    for i,p in enumerate(paragraphs):
        paragraph_text=kpdf.xml_to_raw_text(p)
        #print("paragraph_text: ", paragraph_text)
        
        new_section=get_section(p, paragraph_text, current_section)
        if new_section!=current_section:
            current_section=new_section
            sections.append((current_section, p))
        
        #save to dataframe
        styles_df, paragraph_xml, paragraph_info=process_paragraph_text(p, doc_id, i, paragraph_text, styles_df, usual_font_size, doc_pos)
        #par_fun=get_paragraph_function(paragraph_info, current_section)
        par_fun="Pending"
        if i>0:
            paragraphs_df.loc[len(paragraphs_df)-1, 'next'] = i

        paragraphs_df.loc[len(paragraphs_df)] = [doc_id, i, i-1, -1, len(paragraph_text), doc_pos, current_section, par_fun, paragraph_text.strip("\n")]
        
        doc_xml+="<paragraph>\n"+paragraph_xml+"\n</paragraph>\n"
        doc_pos=doc_pos+len(paragraph_text)
        
    doc_xml+="</document>"

    return styles_df, paragraphs_df, sections, doc_xml


def get_section(paragraph, paragraph_text, current_section):
    
    # #get ratio of uppercase to lowercase
    # uc_proportion, _=krusty.count_uppercase(paragraph_text)
    # #use ratio and length to assess likelihood of title
    # if uc_proportion>0.3 and len()<200:
    #     likely_title=True
    
    # # Antecedentes
    # if likely_title and "antecedentes" in paragraph_text.lower():
    #     current_section="Factual Background"
        
    # elif likely_title and "norma demandada" in paragraph_text.lower():
    #     current_section="Norma demandada"
        
    # elif likely_title and "antecedentes" in paragraph_text.lower():
    #     current_section="Factual Background"
    
    return current_section


def find_section_format(paragraphs_df, styles_df, document_id):
    '''
    Finds the most common format among the paragraphs that include specific keywords,
    and a pattern for that format.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs in the document.
    styles_df : pd.DataFrame
        A dataframe containing information about the styles in the document.
    document_id : int
        The ID of the document.

    Returns
    -------
    most_common_format : str
        The most common format among the paragraphs that include specific keywords.
    pattern : str
        A pattern for the most common format.

    '''

    sections_looked_for=["ANTECEDENTES", "CONSIDERACIONES", "NORMA DEMANDADA", "NORMA ACUSADA", "DEMANDA", "DECISION", "DECISIÓN"]
    res=[]
    for keyword in sections_looked_for:
        #find rows that are in the document and contain the keyword
        matches=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['text'].str.contains(keyword, case=True))]
        print("Number of matches for ", keyword," is ", len(matches))
        #if len(matches)==1:
        if len(matches)>=1: #experimental
            found=False
            for r in res:
                if r['paragraph_id']==matches.iloc[0]['paragraph_id'] and r['document']==matches.iloc[0]['document']:
                    print("paragraph already in res")
                    found=True
                    break

            if not found:
                res.append(matches.iloc[0])
        
    
    #sort res by paragraph_id
    # res.sort(key=lambda x: x['paragraph_id'])

    #print("res", res)
    #find format
    formats=[]
    previous_number=0
    for entry in res:
        #print("entry", entry)
        print("entry id", entry['paragraph_id'])
        print("entry text", entry['text'])
        print("previous paragraph id", entry['previous'])
        piece=entry['text'].strip()[0:15]
        current_format, current_number=krusty.detect_numerals_by_format(piece)

        print("current_format", current_format)
        print("current_number", current_number)

        #previous_number=current_number
        if current_format=="unknown" and entry['previous']!=-1:
            
            #get updated info for entry:
            entry=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['paragraph_id'])].iloc[0]

            #get previous paragraph that matches the document
            previous_par=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['previous'])].iloc[0]

            #check if the previous paragraphs is a number
            previous_par_format, current_number=krusty.detect_numerals_by_format(previous_par['text'].strip())
            print("previous paragraph format", previous_par_format)
            print("previous paragraph text", previous_par['text'])

            if previous_par_format!="unknown":

                #get index of previous paragraph
                index=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==previous_par['paragraph_id'])].index[0] 
                
                #combine the two paragraphs
                paragraphs_df.loc[index, 'text']=previous_par['text']+entry['text']
                paragraphs_df.loc[index, 'length']=previous_par['length']+entry['length']

                #update 'next' of previous paragraph
                paragraphs_df.loc[index, 'next']=entry['next']
                #print("updated previous paragraph", paragraphs_df.loc[index])

                #update 'previous' of new next paragraph
                #get index of next paragraph
                index_next=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['next'])].index[0]
                paragraphs_df.loc[index_next, 'previous']=previous_par['paragraph_id']
                #print("updated next paragraph", paragraphs_df.loc[index_next])

                #modify styles_df to update paragraph_id
                #get rows in styles_df that have the same paragraph_id as the entry
                #ids=styles_df[styles_df['paragraph_id']==entry['paragraph_id']].index
                ids=styles_df[(styles_df['document']==document_id) &(styles_df['paragraph_id']==entry['paragraph_id'])].index

                for id in ids:
                    styles_df.loc[id, 'paragraph_id']=previous_par['paragraph_id']
                    styles_df.loc[id, 'paragraph_pos']=len(previous_par['text'])+styles_df.loc[id, 'paragraph_pos']
                
                #get current paragraph index
                index_current=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['paragraph_id'])].index[0]
                #drop current paragraph
                paragraphs_df.drop(index_current, inplace=True)

                current_format=previous_par_format
                
        formats.append(current_format)
        
    print("Formats before pruning:", formats)
    
    #get most common format among the paragraphs that included the words we were interested in
    # remove all ocurrenences of "unknown"
    formats = [x for x in formats if x != "unknown"]
    print("Formats after pruning:", formats)
    most_common_format=krusty.most_common_item(formats)

    if most_common_format is not None:
        if most_common_format=="ara-hyphen":
            #number at the beginning of sentence that is not followed by a period and that is followed by a hyphen and a space or another letter, but not a number
            pattern = r"^\b\d+\b[ ]*[-][^0-9]"
        elif most_common_format=="ara-dot":
            #number at the beginning of sentence that is not followed by a period and that is followed by a hyphen and a space or and then uppercase letters, but not a number
            pattern = r"^\d+[ ]*[\.][ ]*[-]*[A-Z]{2,}"
        elif most_common_format=="rom-hyphen":
            pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})-"
        elif most_common_format=="rom-dot":
            pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})."
        elif most_common_format=="alpha-dot":
            pattern = r"^[A-Z]{1}\."
        elif most_common_format=="alpha-hyphen":
            pattern = r"^[A-Z]{1}-"
    else:
        most_common_format="unknown"
        pattern = ""
    
    return most_common_format, pattern

def find_sections(paragraphs_df, section_format, numbering_pattern, document_id=-1):
    '''
    Detects paragraphs that match the section format and numbering pattern.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs in the document.
    section_format : str
        The format of the sections to be found.
    numbering_pattern : str
        The numbering pattern of the sections to be found.
    document_id : int, optional
        The ID of the document. If provided, only paragraphs from that document will be considered. 
        The default is -1.

    Returns
    -------
    matching_paragraphs : List[pd.Series]
        A list of paragraphs that match the section format and numbering pattern.

    '''
    if document_id!=-1:
        paragraphs_df=paragraphs_df[paragraphs_df['document']==document_id]

    matching_paragraphs=[]
    print("Section format is ", section_format)
    current_number=-1
    if section_format=="rom-hyphen" or section_format=="rom-dot": #if sections are denoted by roman numerals
        for index, p in paragraphs_df.iterrows():
            # #check if variable is not a String
            # if not isinstance(p['text'], str):
            #     print("Text is not a string: ", p['text'])
            text=p['text'].strip()[0:10]
            text=krusty.clean_up_spaces(text)
            aux=text.split()
            text=aux[0]
            if len(p['text'])<75 and len(aux)>1: #if it's too long, then it shouldn't be a title, if it's only a number, then it is also wrong
                pure, dot, hyphen=krusty.detect_roman_numerals_by_format(text)
                
                if section_format=="rom-hyphen" and len(hyphen)>0:
                    next_up=krusty.next_number(hyphen[0])
                    if next_up>=current_number and next_up-current_number<10:
                        matching_paragraphs.append(p)
                        print(hyphen)
                        current_number=next_up

                elif section_format=="rom-dot" and len(dot)>0:
                    next_up=krusty.next_number(dot[0])
                    if next_up>=current_number and next_up-current_number<10:
                        matching_paragraphs.append(p)
                        print(dot)
                        current_number=next_up
            
    elif section_format=="ara-hyphen" or section_format=="ara-dot":
        for index, p in paragraphs_df.iterrows():
            #print("paragraph text (arabic numeral detection)", p['text'])
            aux_text=p['text'].strip()[0:15]
            #print("aux_text", aux_text)
            aux=re.search(numbering_pattern, aux_text)
            if aux and len(aux_text.split())>0:
                #print match
                print("match", aux.group())
                print("paragraph text", p['text'])
                #get first number
                next_up=krusty.next_number(aux_text.split()[0])
                if next_up>=current_number:
                    matching_paragraphs.append(p)
                    #print(aux)
                    current_number=next_up
#TODO: add alpha-hyphen and alpha-dot? Should we do this?
        
    # Display the matching rows
    for row in matching_paragraphs:
        print("document:", row["document"], "paragraph_id:", row["paragraph_id"], "text:", row['text'], sep=", ")

    
    #sort matching paragraphs by paragraph_id
    matching_paragraphs.sort(key=lambda x: x['paragraph_id'])
        
    return matching_paragraphs
    

def find_paragraphs_matching_format(paragraph, p_format=""):
    return ""
    
def get_paragraph_function(paragraph_info, section):
    
    return "Pending"

def get_xml_from_dataframe(paragraph_df):
    xml="<document>\n"
    
    xml="</document>"
    return xml

def update_document_sections(document_id, paragraphs_df, styles_df, sections, case=False):
    '''
    Updates the section names for the paragraphs and styles in the dataframes and returns the updated dataframes.

    Parameters
    ----------
    document_id : int
        The ID of the document.
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the styles.
    sections : List[Dict[str, Union[int, str]]]
        A list of dictionaries containing information about the sections.

    Returns
    -------
    paragraphs_df : pd.DataFrame
        The updated dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        The updated dataframe containing information about the styles.
    '''

    section_list, section_names_re=get_section_mapping()

    i=0
    while i<len(sections):
        #get section paragraph_id
        section_id=sections[i]['paragraph_id']
        #get section name
        section_name=match_sections_to_names(sections[i]['text'][0:50], case) #only use first 50 characters to match section name, experiment with this
        if i==len(sections)-1:
            next_section_id=float('inf')
        else:
            next_section_id=sections[i+1]['paragraph_id']

        #get indices of paragraphs between the section and the next section that belong to the document
        indices=paragraphs_df[(paragraphs_df['paragraph_id']>=section_id) & (paragraphs_df['paragraph_id']<next_section_id) & (paragraphs_df['document']==document_id)].index
        
        
        #if name is pending, look for a fuzzy match
        if section_name=="Pending":
            #get text of the section
            section_text=sections[i]['text']
            match, associated_terms, min_distance=find_closest_match(section_list, section_names_re, section_text)
            if min_distance<30:
                section_name=match
                print("Section name was found throug fuzzy matching: ", section_name)
                print("Associated terms: ", associated_terms)

        #update the section name for the paragraphs
        paragraphs_df.loc[indices, 'section']=section_name

        #update the section name for the styles
        indices=styles_df[(styles_df['document']==document_id) & (styles_df['paragraph_id']>=section_id) & (styles_df['paragraph_id']<next_section_id)].index
        styles_df.loc[indices, 'section']=section_name

        i+=1    

    #paragraphs_df, styles_df = find_salvamentos(paragraphs_df, styles_df)

    #convert pandas to csv
    #paragraphs_df.to_csv("paragraphs.csv", index=False)

    print("Section names have been updated")
    return paragraphs_df, styles_df

def match_sections_to_names(text, case=False):
    '''
    Takes in a text string and returns the name of the section that matches the text.

    Parameters
    ----------
    text : str
        The text to be matched to a section name.

    Returns
    -------
    res : str
        The name of the section that matches the text.

    '''
    
    sections, section_names_re=get_section_mapping()
    res=krusty.get_mentions(text, sections, section_names_re, case=case)
    
    return res

def update_sections_all_documents(paragraphs_df, styles_df, documents_df):
    '''
    Updates the section names for all documents in the dataframes.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the styles.
    documents_df : pd.DataFrame
        A dataframe containing information about the documents.

    Returns
    -------
    paragraphs_df : pd.DataFrame
        The updated dataframe containing information about the paragraphs in the documents.
    styles_df : pd.DataFrame
        The updated dataframe containing information about the styles in the documents.
    documents_df : pd.DataFrame
        The updated dataframe containing information about the documents.

    '''
    
    #add column to documents_df to store the number of sections if it doesn't exist
    if 'num_sections' not in documents_df.columns:
        documents_df['num_sections']=0

    #get list of unique documents in the dataframe
    documents=paragraphs_df["document"].unique()

    #section names set
    section_names=set()

    #loop through the documents
    for document in documents:
        #get document name
        document_name=documents_df[documents_df['id']==document]['name'].iloc[0]
        print("Updating sections for document ", document, "(", document_name, ")")
        common_format, pattern=find_section_format(paragraphs_df, styles_df, document)
        if common_format!="unknown":
            sections=find_sections(paragraphs_df, common_format, pattern, document)


            #-------------------------------------
            #add section names to set
            new_section_names=get_section_names(sections)
            # print("Section names found: ", new_section_names)
            section_names=section_names.union(new_section_names)
            #-------------------------------------
            
            #update number of sections in documents_df
            documents_df.loc[documents_df['id']==document, 'num_sections']=len(sections)

            paragraphs_df, styles_df=update_document_sections(document, paragraphs_df, styles_df, sections)
        else:
            print("No sections following a number format found. Attempting to find sections using other format info...")

            sections_without_numbers=get_sections_without_number(document, paragraphs_df, styles_df)
            new_section_names=get_section_names(sections_without_numbers)
            section_names=section_names.union(new_section_names)
            if len(sections_without_numbers)>0:
                #update number of sections in documents_df
                documents_df.loc[documents_df['id']==document, 'num_sections']=len(sections_without_numbers)

                paragraphs_df, styles_df=update_document_sections(document, paragraphs_df, styles_df, sections_without_numbers, case=True)
            else:
                print("No sections found for document ", document)
        if len(new_section_names)>0:
            print("Section names found: ", new_section_names)
        print("-------------------------------------------------------------------------------")
    
    paragraphs_df, styles_df = find_salvamentos(paragraphs_df, styles_df)

    for s in section_names:
        print(s)

    set_number_of_paragraphs_per_document(documents_df, paragraphs_df)

    detect_signatures(paragraphs_df, styles_df)

    return paragraphs_df, styles_df, documents_df

def get_document_statistics(paragraphs_df, styles_df, document_id, verbose=False):
    '''
    Gets statistics about the text types and their lengths for each section in the document.
    
    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs in the document.
    styles_df : pd.DataFrame
        A dataframe containing information about the styles in the document.
    document_id : int
        The ID of the document.

    Returns
    -------
    res_df : pd.DataFrame
        A dataframe with statistics about the text types and their lengths for each section in the document.

    '''
    #get unique sections in the document
    sections=paragraphs_df[paragraphs_df['document']==document_id]['section'].unique()
    for section in sections:
        #get all rows in styles_df that belong to the document and the section
        section_df=styles_df[(styles_df['document']==document_id) & (styles_df['section']==section)]
        #get all different text types in the section
        text_types=section_df['text_type'].unique()
        #get total length for each text type
        for text_type in text_types:
            length=section_df[section_df['text_type']==text_type]['length'].sum()
            print("Document:", document_id, "Section:", section, "Text type:", text_type, "Length:", length) if verbose==True else ""

    res_df=styles_df.groupby(['document','section', 'text_type']).agg({'length': 'sum'}).reset_index()
    
    return res_df

def get_sections_statistics(paragraphs_df, styles_df, verbose=False):
    '''
    Gets statistics about the text types and their lengths for each section in all documents.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs in the documents.
    styles_df : pd.DataFrame
        A dataframe containing information about the styles in the documents.

    Returns
    -------
    res_df : pd.DataFrame
        A dataframe with statistics about the text types and their lengths for each section in all documents.

    '''
    #get unique documents in the dataframe
    documents=paragraphs_df['document'].unique()
    #create dataframe to store results
    res_df=pd.DataFrame(columns=['document', 'section', 'text_type', 'length'])
    #loop through the documents
    for document in documents:
        cur_res_df=get_document_statistics(paragraphs_df, styles_df, document, verbose)
        #get year from documents_df
        #merge with results dataframe
        res_df=pd.concat([res_df, cur_res_df], ignore_index=True)
    return res_df


def time_analysis(documents_df, stats_df, type="X"):
    '''
    Calculates average lengths of sections and text types for each year

    Parameters
    ----------
    documents_df : pd.DataFrame
       A dataframe containing information about the documents.
   stats_df : pd.DataFrame
       A dataframe containing statistics about the sections and text types in the documents.
   type : str, optional
       The type of documents to be analyzed. If not provided, all documents will be analyzed 
       The default is "X".

    Returns
    -------
    section_res_df : pd.DataFrame
        A dataframe with average lengths of sections for each year.
    sec_text_types_res_df : pd.DataFrame
        A dataframe with average lengths of text types for each section and year.

    '''
    
    
    if type!="X":
        docs_df=documents_df[documents_df['type']==type].copy()
        print(len(docs_df), " documents of type ", type, " found")
    else:
        docs_df=documents_df.copy()
        print("Analyzing all ". len(docs_df), " documents...")
    #get unique text types
    text_types=stats_df['text_type'].unique()
    #get unique years
    years=docs_df['year'].unique()
    #get unique sections
    sections=stats_df['section'].unique()
    #sort years in ascending order
    years.sort()
    #create dataframe to store results
    section_res_df=pd.DataFrame(columns=['year', 'section', 'avg_length'])
    sec_text_types_res_df=pd.DataFrame(columns=['year', 'section', 'text_type', 'avg_length'])
    #loop through the years
    for year in years:
        #get document ids for entries for the year
        doc_ids_year=docs_df[docs_df['year']==year]['id'].unique()
        #get average lenght for each section for the year
        for section in sections:
            if section!="Pending":
                #get average length for the year and section
                avg_length=stats_df[(stats_df['document'].isin(doc_ids_year)) & (stats_df['section']==section)]['length'].mean()
                #if average length is NaN, set it to 0
                if np.isnan(avg_length):
                    avg_length=0
                print("Year:", year, "Section:", section, "Average length:", avg_length)
                #save to dataframe
                section_res_df.loc[len(section_res_df)] = [year, section, avg_length]
                #get average length for each text type for the year and section
                for text_type in text_types:
                    avg_length=stats_df[(stats_df['document'].isin(doc_ids_year)) & (stats_df['section']==section) & (stats_df['text_type']==text_type)]['length'].mean()
                    #if average length is NaN, set it to 0
                    if np.isnan(avg_length):
                        avg_length=0
                    print("Year:", year, "Section:", section, "Text type:", text_type, "Average length:", avg_length)
                    #save to dataframe
                    sec_text_types_res_df.loc[len(sec_text_types_res_df)] = [year, section, text_type, avg_length]

    return section_res_df, sec_text_types_res_df


def detect_verbs(text):
    verbs = []
    nlp = spacy.load("es_core_news_sm")
    #nlp = spacy.load("es_core_web_lg")

    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append(token.text)
    return verbs


def get_word_frequencies(blocks_df, text_type="quote", verbose=False):
    '''
    Provides lists of verbs, positive words, and negative words found in the blocks.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        A dataframe containing information about the blocks.
    verbose : bool, optional
        If True, the function will print additional information.
        The default is False.

    Returns
    -------
    verbs : List[str]
        A list of verbs found in the blocks.
    positive_words : List[str]
        A list of positive words found in the blocks.
    negative_words : List[str]
        A list of negative words found in the blocks.

    '''
    
    verbs=[]
    positive_words=[]
    negative_words=[]
    #find all blocks matching the text type
    matched_blocks=blocks_df[blocks_df['text_type']==text_type]
    #for block in quote_blocks:
    for index, block in matched_blocks.iterrows():
        print("current block:","\n", block['text'])  if verbose==True else ""
        #find previous row that belongs to the same paragraph using paragraph position
        aux_previous=blocks_df[(blocks_df['paragraph_id']==block['paragraph_id']) & (blocks_df['paragraph_pos']<block['paragraph_pos'])]
        if len(aux_previous)>0:
            #get previous block
            previous_block=aux_previous.loc[aux_previous['paragraph_pos'].idxmax()]
            print("previous_block:", "\n", previous_block['text'])  if verbose==True else ""
            previous_block_verbs=detect_verbs(previous_block['text'])  if verbose==True else ""
            print("previous verbs", previous_block_verbs)
            verbs=verbs+previous_block_verbs

    return verbs, positive_words, negative_words


def verb_analysis(blocks_df, text_type):
    '''
    Takes blocks and text type and returns information about the verbs found in the blocks of the specified text type.
    Gets a dataframe with statistics about the verbs.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        A dataframe containing information about the blocks.
    text_type : str
        The text type to be analyzed.

    Returns
    -------
....verb_dictionary : Dict[str, Dict[str, Union[int, Dict[str, int]]]]
        A dictionary with information about the verbs found in the blocks of the specified text type.
    verbs_df : pd.DataFrame
        A dataframe with statistics about the verbs found in the blocks of the specified text type.
    '''
    
    verb_dictionary = {}
    nlp = spacy.load("es_core_news_sm")
    
    blocks=blocks_df[blocks_df['text_type']==text_type]
    for index, block in blocks.iterrows():
        print("current block:","\n", block['text'])
        doc = nlp(block['text'])
        for token in doc:
            if token.pos_ == "VERB":
                #if verb already exists in dictionary
                if token.lemma_ in verb_dictionary:
                    #increment counter overall frequency
                    verb_dictionary[token.lemma_]['frequency']= verb_dictionary[token.lemma_]['frequency']+1
                    #check if conjugation already exists
                    if token.text in verb_dictionary[token.lemma_]['conjugations']:
                        #increment counter for conjugation
                        verb_dictionary[token.lemma_]['conjugations'][token.text]=verb_dictionary[token.lemma_]['conjugations'][token.text]+1
                    else:
                        #create new entry in conjugations dictionary
                        verb_dictionary[token.lemma_]['conjugations'][token.text]=1

                else:
                    #create new entry in dictionary
                    verb_dictionary[token.lemma_]={'frequency':1, 'conjugations':{token.text: 1}}
    #create dataframe from dictionary

    verbs_df=pd.DataFrame(columns=['verb', 'frequency', 'most_frequent_conjugation'])
    for verb in verb_dictionary:
        verbs_df.loc[len(verbs_df.index)] = [verb, verb_dictionary[verb]['frequency'], max(verb_dictionary[verb]['conjugations'], key=verb_dictionary[verb]['conjugations'].get)]

    return verb_dictionary, verbs_df

def verb_analysis2(blocks_df, text_type):
    '''
    Obtains info about the verbs found in the blocks of the specified text type and statistics about verbs and phrases
    surrounding the verbs.

    Parameters
    ----------
    blocks_df : pd.DataFrame
        A dataframe containing information about the blocks.
    text_type : str
        The text type to be analyzed.

    Returns
    -------
    verb_dictionary : Dict[str, Dict[str, Union[int, Dict[str, int]]]]
        A dictionary with information about the verbs found in the blocks of the specified text type.
    verbs_df : pd.DataFrame
        A dataframe with statistics about the verbs found in the blocks of the specified text type.
    phrases_df : pd.DataFrame
        A dataframe with statistics about phrases surrounding the verbs found in the blocks of the specified text type.

    '''
    
    verb_dictionary = {}
    nlp = spacy.load("es_core_news_sm")
    
    phrases_df=pd.DataFrame(columns=['phrase', 'frequency', 'block_id'])

    blocks=blocks_df[blocks_df['text_type']==text_type]
    for index, block in blocks.iterrows():
        #print("current block:","\n", block['text'])
        doc = nlp(block['text'])
        for token in doc:
            if token.pos_ == "VERB":

                #find text surrounding verb
                verb_pos=block['text'].find(token.text)
                previous_text=block['text'][:verb_pos].strip()
                previous_word_beginning=previous_text.rfind(" ")

                phrase=previous_text[previous_word_beginning+1:]+" "+token.text
                print("phrase:", phrase)
                if phrase in phrases_df['phrase'].values:
                    phrases_df.loc[phrases_df['phrase']==phrase, 'frequency']=phrases_df.loc[phrases_df['phrase']==phrase, 'frequency']+1
                else:
                    phrases_df.loc[len(phrases_df.index)] = [phrase, 1, block['id']]
                

                if token.lemma_ in verb_dictionary: #if verb already exists in dictionary
                    #increment counter overall frequency
                    verb_dictionary[token.lemma_]['frequency']= verb_dictionary[token.lemma_]['frequency']+1
                    #check if conjugation already exists
                    if token.text in verb_dictionary[token.lemma_]['conjugations']:
                        #increment counter for conjugation
                        verb_dictionary[token.lemma_]['conjugations'][token.text]=verb_dictionary[token.lemma_]['conjugations'][token.text]+1
                    else:
                        #create new entry in conjugations dictionary
                        verb_dictionary[token.lemma_]['conjugations'][token.text]=1

                else:
                    #create new entry in dictionary
                    verb_dictionary[token.lemma_]={'frequency':1, 'conjugations':{token.text: 1}}
        
        
    #create dataframe from dictionary
    verbs_df=pd.DataFrame(columns=['verb', 'frequency', 'most_frequent_conjugation'])
    for verb in verb_dictionary:
        verbs_df.loc[len(verbs_df.index)] = [verb, verb_dictionary[verb]['frequency'], max(verb_dictionary[verb]['conjugations'], key=verb_dictionary[verb]['conjugations'].get)]

    return verb_dictionary, verbs_df, phrases_df

def verb_analysis_by_text_type(blocks_df, verbose=False):
    """
    

    Parameters
    ----------
    blocks_df : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    verb_dictionary : Dictionary
        DESCRIPTION.
    verbs_df : Pandas DataFrame
        DESCRIPTION.
    phrases_df : Pandas DataFrame
        DESCRIPTION.
    phrases_details_df : Pandas DataFrame
        DESCRIPTION.

    """
    
    verb_dictionary = {}
    nlp = spacy.load("es_core_news_sm")
    
    phrases_df=pd.DataFrame(columns=['phrase', 'frequency'])
    phrases_details_df=pd.DataFrame(columns=['phrase', 'block_id', "where"])

    #TODO If we sort by document, paragraph and paragraph_pos, we can avoid having to sort by paragraph_pos in the loop
    #sort blocks by document, paragraph and paragraph_pos
    blocks_df=blocks_df.sort_values(by=['document', 'paragraph_id', 'paragraph_pos'])
    display(blocks_df)
    for index, block in blocks_df.iterrows():
        blocks=blocks_df[(blocks_df['document']==block['document']) & (blocks_df['paragraph_id']==block['paragraph_id'])]
        
        where="pending"
        #get previous block type
        #previous_block_type=blocks_df[blocks_df['id']==block['id']-1]['text_type'].values[0] #Maybe, if we can guarantee block numbering integrity...
        aux_previous=blocks[(blocks['paragraph_pos']<block['paragraph_pos'])]
        #get previous block
        if len(aux_previous)>0:
            #get the block for which the paragraph_pos is the highest    
            # previous_block=aux_previous[aux_previous['paragraph_pos']==aux_previous['paragraph_pos'].max()]
            # previous_block_type=previous_block['text_type'].values[0]

            previous_block=aux_previous[aux_previous['paragraph_pos']==aux_previous['paragraph_pos'].max()].iloc[0]
            previous_block_type=previous_block['text_type']

            
        else:
            previous_block_type="none"

        #get next block type
        aux_next=blocks[(blocks['paragraph_pos']>block['paragraph_pos'])]
        #get next block
        if len(aux_next)>0:
            #get the block for wich the paragraph_pos is the lowest    
            next_block=aux_next[aux_next['paragraph_pos']==aux_next['paragraph_pos'].min()]
            next_block_type=next_block['text_type'].values[0]
        else:
            next_block_type="none"

        where=block['text_type']
        
        if previous_block_type=='quote':
            where=where + " after quote"
        
        if next_block_type=='quote':
            where=where + " before quote"
        
        print("current block:","\n", block['text']) if verbose==True else ""
        if block['text']==np.nan:
            block['text']=""

        try:
            doc = nlp(block['text'])
            for token in doc:
                if token.pos_ == "VERB":
                    #find text surrounding verb
                    verb_pos=block['text'].find(token.text)
                    previous_text=block['text'][:verb_pos].strip()
                    previous_word_beginning=previous_text.rfind(" ")
                    if previous_word_beginning==-1:
                        phrase=token.text
                    else:
                        phrase=previous_text[previous_word_beginning+1:]+" "+token.text

                    print("phrase:", phrase) if verbose==True else ""
                    
                    if phrase in phrases_df['phrase'].values:
                        phrases_df.loc[phrases_df['phrase']==phrase, 'frequency']=phrases_df.loc[phrases_df['phrase']==phrase, 'frequency']+1
                    else:
                        phrases_df.loc[len(phrases_df.index)] = [phrase, 1]
                    
                    phrases_details_df.loc[len(phrases_details_df.index)] = [phrase, block['id'], where]
                    
                    if token.lemma_ in verb_dictionary: #if verb already exists in dictionary
                        #increment counter overall frequency
                        verb_dictionary[token.lemma_]['frequency']= verb_dictionary[token.lemma_]['frequency']+1
                        #check if conjugation already exists
                        if token.text in verb_dictionary[token.lemma_]['conjugations']:
                            #increment counter for conjugation
                            verb_dictionary[token.lemma_]['conjugations'][token.text]=verb_dictionary[token.lemma_]['conjugations'][token.text]+1
                        else:
                            #create new entry in conjugations dictionary
                            verb_dictionary[token.lemma_]['conjugations'][token.text]=1
                    else:
                        #create new entry in dictionary
                        verb_dictionary[token.lemma_]={'frequency':1, 'conjugations':{token.text: 1}}
        except:
            print("Error processing block: ", block['text'])
            print("Previous block type: ", previous_block_type)
            print("Next block type: ", next_block_type)
            print("Where: ", where)
            print("--------------------------------------------------------------------------------------------------------------------")
            #continue
        
    #create dataframe from dictionary
    verbs_df=pd.DataFrame(columns=['verb', 'frequency', 'most_frequent_conjugation'])
    for verb in verb_dictionary:
        verbs_df.loc[len(verbs_df.index)] = [verb, verb_dictionary[verb]['frequency'], max(verb_dictionary[verb]['conjugations'], key=verb_dictionary[verb]['conjugations'].get)]

    return verb_dictionary, verbs_df, phrases_df, phrases_details_df




def assign_text_function(paragraphs_df, styles_df, document_id, phrases_df, source_folder):

    # get document paragraphs
    document_paragraphs=paragraphs_df[paragraphs_df['document']==document_id]
    for index, paragraph in document_paragraphs.iterrows():
        #get blocks for the paragraph
        paragraph_blocks=styles_df[styles_df['paragraph_id']==paragraph['paragraph_id']]
        #get phrases for the paragraph
        paragraph_phrases=phrases_df[phrases_df['block_id'].isin(paragraph_blocks['id'])]
        
    return ""


def get_verb_stats(phrases_details_df, blocks_df, sections=[], where_scope="targeted", target_folder="verbs_csv"):

    if len(sections)==0:
        #get unique sections
        sections=blocks_df['section'].unique()
    #get blocks for each section
    for section in sections:
        blocks_ids=blocks_df[blocks_df['section']==section]['id'].unique()
        #get phrases for each section
        section_phrases=phrases_details_df[phrases_details_df['block_id'].isin(blocks_ids)]
        if where_scope=="targeted":
            where=["before quote", "after quote"]
        else:
            #get unique where
            where=section_phrases['where'].unique()
        #get phrases for each where
        for w in where:
            print("Section:", section, "Where:", w)
            #aux_df=section_phrases[section_phrases['where']==w]
            #select phrases that contain the where
            aux_df=section_phrases[section_phrases['where'].str.contains(w)]
            aux_df=aux_df.groupby(['phrase']).size().reset_index(name='counts').sort_values(by=['counts'],ascending=False)
            display(aux_df)
            #save to csv, if folder doesn't exist, create it
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            aux_df.to_csv(target_folder+"/"+section+"_"+w+".csv", index=False)

def verb_pruning(source_folder="verbs_csv", target_folder="verbs_csv_pruned", min_frequency=5):
    '''
    Takes in source and target folder names, and a minimum frequency. 
    It reads CSV files from the source folder, prunes the data based on the minimum frequency, and saves the pruned data to the target folder.


    Parameters
    ----------
    source_folder : str, optional
        The name of the source folder. The default is "verbs_csv".
    target_folder : str, optional
        The name of the target folder. The default is "verbs_csv_pruned".
    min_frequency : int, optional
        The minimum frequency for a verb to be included in the pruned data. The default is 5.

    Returns
    -------
    None.

    '''
    nlp = spacy.load("es_core_news_sm")

    #get files in folder
    files=os.listdir(source_folder)
    #loop through files
    for f in files:
        print("Processing file: ", f)
        #check that file is csv
        if f.endswith(".csv"):
            #read csv
            df=pd.read_csv(source_folder+"/"+f)

            #look for subjunctives
            subjunctives_df=df.copy()
            subjunctives_df=subjunctives_df[(subjunctives_df['phrase'].str.startswith('se ')) | (subjunctives_df['phrase'].str.startswith('Se '))]
            #remove subjunctives from original df
            df=df[~df['phrase'].isin(subjunctives_df['phrase'])]
            #convert prhases to lowercase
            subjunctives_df['phrase']=subjunctives_df['phrase'].str.lower()
            #combine same phrases by adding their counts
            subjunctives_df=subjunctives_df.groupby(['phrase']).agg({'counts': 'sum'}).reset_index()
            subjunctives_df=subjunctives_df.sort_values(by=['counts'], ascending=False)
            # add function column
            subjunctives_df['function']='subjunctive'
            display(subjunctives_df)

            #look for modal verbs
            #modal_verbs_df=df.copy()
            #modal_verbs_df=modal_verbs_df[(modal_verbs_df['phrase'].str.startswith('debe ')) | (modal_verbs_df['phrase'].str.startswith('Debe ')) | (modal_verbs_df['phrase'].str.startswith('deberá ')) | (modal_verbs_df['phrase'].str.startswith('Deberá '))]
            
            #look for two successive verbs
            #create a dataframe with fields phrase and counts
            two_verbs_df=pd.DataFrame(columns=['phrase', 'counts'])
            #loop through phrases
            for index, row in df.iterrows():
                #count words
                words=row['phrase'].split()
                #if there are two words
                if len(words)==2:
                    #is the first word a verb?
                    doc = nlp(row['phrase'])
                    for ind2, token in enumerate(doc):
                        if ind2==0 and token.pos_ == "VERB":
                            #add to dataframe
                            two_verbs_df.loc[len(two_verbs_df.index)] = [row['phrase'], row['counts']]

            #remove two verbs from original df
            df=df[~df['phrase'].isin(two_verbs_df['phrase'])]
            #combine same phrases by adding their counts
            two_verbs_df=two_verbs_df.groupby(['phrase']).agg({'counts': 'sum'}).reset_index()
            two_verbs_df=two_verbs_df.sort_values(by=['counts'], ascending=False)
            # add function column
            two_verbs_df['function']='two verbs'
            display(two_verbs_df)
            
            #save to csv
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            subjunctives_df.to_csv(target_folder+"/"+f.replace(".csv", "")+"_subjunctives.csv", index=False)
            two_verbs_df.to_csv(target_folder+"/"+f.replace(".csv", "")+"_two_verbs.csv", index=False)


def get_word_cloud_from_df(df, column_name, stopwords=[], max_words=100, max_font_size=50, background_color="black", width=800, height=400):
    #convert df to dict
    word_dict = df.to_dict()[column_name]
    #generate word cloud
    # Create a dictionary from the DataFrame
    word_freq_dict = dict(zip(df['word'], df['frequency']))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

    wordcloud = WordCloud(stopwords=stopwords, max_words=max_words, max_font_size=max_font_size, background_color=background_color, width=width, height=height).generate_from_
    #plot word cloud
    plt.imshow(wordcloud, interpolation='bilinear')


def get_file_metadata_from_db(filename, conn=None):
    
    #remove extension from filename if there is more than one dot
    extension_index=filename.rfind(".")
    if extension_index!=-1:
        filename=filename[:extension_index]
        #print("filename: ", filename)
    if conn is None:
        conn=krusty.connect_to_db("cc.db", "cases")
    #select file that starts with filename
    query="SELECT * FROM cases WHERE filename LIKE '"+filename+"%'"
    c=conn.cursor()
    c.execute(query)
    r=c.fetchone()
    return r, conn

def find_salvamentos(paragraphs_df, styles_df):

    paragraphs_df=paragraphs_df.copy()
    styles_df=styles_df.copy()

    #set column

    salvamentos=styles_df[ ( (styles_df['text_type']=='emphasis') | (styles_df['text'].str.isupper()) ) 
                          & (~styles_df['text'].str.match(r'^\d+', na=False))
                          & ((styles_df['text'].str.contains("salvamento", na=False, case=False)) | (styles_df['text'].str.contains("aclaración de voto", na=False, case=False))) 
                          & (~styles_df['text'].str.contains("motivo", na=False, case=False)) 
                          & (~styles_df['text'].str.contains("fundamentos", na=False, case=False))
                          & (~styles_df['text'].str.contains("razones", na=False, case=False))
                          & (~styles_df['text'].str.contains("razón", na=False, case=False))]

    current_document=0
    current_opinion=0
    for index, row in salvamentos.iterrows():

        if row['document']!=current_document:
            current_document=row['document']
            current_opinion=1

        paragraph=paragraphs_df[ (paragraphs_df['document']==row['document']) & (paragraphs_df['paragraph_id']==row['paragraph_id']) ]
        #get paragraph index
        paragraph_index=paragraph.index[0]

        if len(paragraph['text'].iloc[0])<len(row['text'])+100:
            #change section to "Salvamento de voto"
            paragraphs_df.loc[paragraph_index, 'section']="Dissenting Opinion - " + str(current_opinion)
            current_opinion=current_opinion+1

            #update corresponding styles
            for ind_block, block in styles_df[(styles_df['document']==row['document']) & (styles_df['paragraph_id']==row['paragraph_id'])].iterrows():
                styles_df.loc[ind_block, 'section']="Dissenting Opinion - " + str(index)




    paragraphs_df['opinion'] = paragraphs_df.section.str.extract('(\d)')
    paragraphs_df['aux_section'] = paragraphs_df['section'].copy()

    sections=paragraphs_df['section'].unique()
    #convert to pandas series
    sections=pd.Series(sections)
    #get all sections that do not contain the text "Dissenting Opinion"
    sections=sections[~sections.str.contains("Dissenting Opinion")]
    for s in sections:
        paragraphs_df['section']=paragraphs_df['section'].replace(s, np.nan)

    paragraphs_df=paragraphs_df.groupby(by='document', group_keys=False
                        ).apply(fill_opinion         # propagate dissenting opinions
                        #).fillna('Decision'          # fill rows prior to dissenting opinion in each document
                        #).replace(np.nan, paragraphs_df['aux_section']
                        #).apply(fill_missing_section, axis=1         # propagate dissenting opinions
                        ).reset_index(drop=True      # clean-up index
                        ).drop(columns = 'opinion')  #remove temporary column
    #paragraphs_df['section'].replace({pd.NA: paragraphs_df['aux_section']}, inplace=True)

    paragraphs_df['section'] = np.where(paragraphs_df['section'].isna(), paragraphs_df['aux_section'], paragraphs_df['section'])
    #drop aux_section column
    paragraphs_df=paragraphs_df.drop(columns=['aux_section'])

    #-------------------DO NOT DELETE----------------
    # paragraphs_df['opinion'] = paragraphs_df.section.str.extract('(\d)')
    # paragraphs_df=paragraphs_df.groupby(by='document', group_keys=False
    #                     ).apply(fill_opinion         # propagate dissenting opinions
    #                     ).fillna('Decision'          # fill rows prior to dissenting opinion in each document
    #                     ).reset_index(drop=True      # clean-up index
    #                     ).drop(columns = 'opinion')  #remove temporary column
    #------------------------------------------------

    # #update styles_df with new sections (dissenting opinions)
    # styles_df['opinion'] = styles_df.section.str.extract('(\d)')
    # styles_df=styles_df.groupby(by='document', group_keys=False
    #                     ).apply(fill_opinion         # propagate dissenting opinions
    #                     ).fillna('Decision'          # fill rows prior to dissenting opinion in each document
    #                     ).reset_index(drop=True      # clean-up index
    #                     ).drop(columns = 'opinion')  #remove temporary column
    
    #drop section column
    styles_df=styles_df.drop(columns=['section'])
    #assign paragraph's section to styles that match document and paragraph_id
    styles_df=styles_df.merge(paragraphs_df[['document', 'paragraph_id', 'section']], on=['document', 'paragraph_id'], how='left')

    return paragraphs_df, styles_df


def fill_opinion(df_d):
    '''
    Fill "dissenting opinions" for one document
    '''   
    # You may replace "section2" with "section" to directly overwrite the section column. Here as a new column for demo.
    
    return df_d.assign(section = 'Dissenting Opinion - ' + df_d['opinion']).ffill()

def fill_missing_section(row):
    if pd.isna(row['section']):
        return row['aux_section']
    else:
        return row['section']

def get_section_names(section_rows):
    section_names=[]
    #iterate through list of rows
    for row in section_rows:
        section_names.append(row['text'])
    return section_names

def check_section_integrity(paragraphs_df, styles_df, document_id):

    #get sections for document
    #document_sections=paragraphs_df[paragraphs_df['document']==document_id]['section'].unique()

    #get the first row for each section
    section_rows=paragraphs_df[paragraphs_df['document']==document_id].groupby('section').first().reset_index().sort_values(by=['paragraph_id'])
    
    display(section_rows) #testing purposes

    #get tuple of text and section
    section_tuples=section_rows[['text', 'section']].itertuples(index=False, name=None)
    section_numbers=[]
    romans=False

    for t in section_tuples:
        aux=t[0].strip().split(" ") #the first word should be the number
        if len(aux[0])>4: #if it's too long, then it's not a number
            print(aux[0])
            min_dot=t[0].find(".")
            min_dash=t[0].find("-")
            if min_dot==-1:
                min_dot=1000
            if min_dash==-1:
                min_dash=1000
            threshold=min(min_dot, min_dash) #the minimum between the first occurrence of dot and dash should be the end of the number
            print("threshold: ", threshold)
            #number=t[0][:threshold].replace(".", "").replace("-", "")
            number=t[0][:threshold]
            print("number: ", number)
        else:
            number=aux[0].replace(".", "").replace("-", "")
        if number.isnumeric()==False:
            romans=True
            print("Roman numeral detected: ", number)
            num=krusty.roman_to_int(number.strip())
            print("Roman numeral converted to: ", num)
            if num!=-2:
                section_numbers.append(num)
            elif t[1]=="header":
                section_numbers.append(0)
        else:
            section_numbers.append(int(number))  

    print(section_numbers)

    #get smallest number
    smallest_number=min(section_numbers)
    #get largest number
    largest_number=max(section_numbers)

    missing_sections=[]
    #check that all integers in between are part of the list
    for i in range(smallest_number, largest_number+1):
        if i not in section_numbers:
            print("Section number ", i, " is missing")
            missing_sections.append(i)

    #look for missing sections in paragraphs_df
    #get paragraphs for document
    document_paragraphs=paragraphs_df[paragraphs_df['document']==document_id]
    if len(missing_sections)>0:
        print("Missing sections: ", missing_sections)
        for s in missing_sections:
            if romans==True:
                #convert to roman numeral
                number=krusty.int_to_roman(s)
            else:
                number=str(s)
            missing_sections_df=document_paragraphs[document_paragraphs['text'].str.contains("^ *"+number, regex=True)]

            #print the text of each paragraph
            for index, row in missing_sections_df.iterrows():
                print("Text: ", row['text'])
                aux_row_text=row['text'].strip().replace(".", "").replace("-", "").replace(" ", "").replace(number, "")
                if len(aux_row_text)==0: #if the text is only the number, then we should merge it with the next paragraph
                    print("Merging with next paragraph")
                    entry=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==row['next'])].iloc[0]


                    #get previous paragraph that matches the document
                    previous_par=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['previous'])].iloc[0]
                    previous_par=row

                    #combine the two paragraphs
                    index=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==previous_par['paragraph_id'])].index[0] 
                    paragraphs_df.loc[index, 'text']=previous_par['text']+entry['text']

                    #update 'next' of previous paragraph
                    paragraphs_df.loc[index, 'next']=entry['next']

                    #update 'previous' of new next paragraph
                    #get index of next paragraph
                    index_next=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['next'])].index[0]
                    paragraphs_df.loc[index_next, 'previous']=previous_par['paragraph_id']

                    #modify styles_df to update paragraph_id
                    #get rows in styles_df that have the same paragraph_id as the entry
                    ids=styles_df[(styles_df['document']==document_id) &(styles_df['paragraph_id']==entry['paragraph_id'])].index

                    for id in ids:
                        styles_df.loc[id, 'paragraph_id']=previous_par['paragraph_id']
                        styles_df.loc[id, 'paragraph_pos']=len(previous_par['text'])+styles_df.loc[id, 'paragraph_pos']


                    
                    #get current paragraph index
                    index_current=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['paragraph_id'])].index[0]
                    #drop current paragraph
                    paragraphs_df.drop(index_current, inplace=True)

                
    else:
        print("No missing sections")

def check_section_integrity_by_document(paragraphs_df, styles_df):
    #get unique documents
    documents=paragraphs_df['document'].unique()
    for d in documents:
        print("Checking document: ", d)
        check_section_integrity(paragraphs_df, styles_df, d)

def get_sections_without_number(document_id, paragraphs_df, styles_df):
    
    _, section_names_re=get_section_mapping()
    #extract all the elements in the list of lists into a single list and convert to uppercase
    sections_looked_for=[item for sublist in section_names_re for item in sublist]
    sections_looked_for=[x.upper() for x in sections_looked_for]

    res=[]
    for keyword in sections_looked_for:
        #find rows that are in the document and contain the keyword
        matches=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['text'].str.contains(keyword, case=True))]
        print("Number of matches for ", keyword," is ", len(matches))
        #if len(matches)==1:
        if len(matches)>=1: #experimental
            found=False
            for r in res:
                if r['paragraph_id']==matches.iloc[0]['paragraph_id'] and r['document']==matches.iloc[0]['document']:
                    print("paragraph already in res")
                    found=True
                    break

            if not found:
                #get correspnding styles
                styles=styles_df[(styles_df['document']==matches.iloc[0]['document']) & (styles_df['paragraph_id']==matches.iloc[0]['paragraph_id'])]
                #check if all styles are of type "emphasis"
                if len(styles[styles['text_type']!='emphasis'])==0:
                        res.append(matches.iloc[0])

    return res


def get_section_mapping():

    sections=["Factual Background",
              
              "Prior Rulings",

              "Court's Analysis",
              
              "Challenged Statute(s)",
              
              "Plaintiff's Allegations",
              
              "Decision",
              
              "Third-Party Intervention",
              
              "Summary of Findings",
              
              "Exhibits",
              
              "Third-Party Intervention: Procurador",

              "Jurisdiction",

              "Hearings"
              ]
     
    section_names_re=[["antecedente", "hechos", "informaci.n preliminar"],
                       
                      ["decisiones objeto de", "decisi.n objeto de", "sentencia revisada", "sentencias revisadas", "fallos que se revisan", 
                       "sentencias objeto de revisi.n", "actuaci.n judicial","actuaciones objeto de deci.n", 
                       "tr.mite surtido en sede de revisi.n", "fallo materia de revisi.n", "primera instancia", "actuaci.n procesal",
                       "tr.mite legislativo y actuaci.n de la corte", "caso concreto", "casos concretos", "segunda instancia", 
                       "actuaci.on procesal", "sentencia que se revisa", "actuaciones en sede de revisi.n", "decisiones judiciales que se revisa"],

                      ["consideraciones", "fundamentos", "fundamento jur.dico", "revisi.n por la corte constitucional"],
                      
                      ["norma acusada", "norma demandada", "normas acusadas", "normas demandadas", "texto del decreto", "texto de la ley", 
                       "norma bajo examen", "disposici.n acusada", "del texto legal objeto", "norma objeto", "ordenamiento acusado",
                       "texto de la", "texto del", "textos legales", "texto revisado"],
                      
                      ["la demanda", "demanda", "los cargos", "impugnaci.n"],
                      
                      ["decisi.n", "d e c i s i . n"], 
                      
                      ["intervenciones", "intervenci.n de terceros", "intervenci.n"],
                      
                      ["s.ntesis del fallo", "conclusi.n"],
                      
                      ["pruebas", "material probatorio aportado", "prueba practicada", "material probatorio"],
                      
                      ["procurador", "procuradur", "ministerio p.blico", "concepto fiscal", "naci.n"],

                      ["competencia"],

                      ["audiencia"]
                      ]
    
    return sections, section_names_re

def find_closest_match(terms, associated_vocab, word):

    i=0
    term_distances=[]
    while i <len(terms):
        associated_distances=[]
        j=0
        while j<len(associated_vocab[i]):
            associated_distances.append(Levenshtein.distance(word, associated_vocab[i][j]))
            j=j+1
        min_distance=min(associated_distances)
        #min_index=associated_distances.index(min_distance)
        term_distances.append(min_distance)
        i=i+1
    min_distance=min(term_distances)
    min_index=term_distances.index(min_distance)
    match=terms[min_index]
    return match, associated_vocab[min_index], min_distance


def fuzzy_section_mapping(df):

    #get section mapping
    sections, section_names_re=get_section_mapping()
    
    #find the first occurrence of "pending" in the section column after a different section
    pending_df=df[df['section'].str.contains("pending", case=False) & (df['section'].shift(1)!=df['section'])]

    for index, row in pending_df.iterrows():
        #find match
        section, matched_term=find_closest_match(sections, section_names_re, row['text'])
        print("Text: ", row['text'])
        print("Matched term: ", section)
        print("Matched terms: ", matched_term)
    
    #to all pending sections, assign the section the previous section
    

def set_number_of_paragraphs_per_document(documents_df, paragraphs_df):
    #add column to documents_df to store the number of paragraphs if it doesn't exist
    if 'num_paragraphs' not in documents_df.columns:
        documents_df['num_paragraphs']=0
    #get unique document ids    
    document_ids=documents_df['id'].unique()
    #loop through document ids
    for document_id in document_ids:
        #get number of paragraphs for document
        num_paragraphs=len(paragraphs_df[paragraphs_df['document']==document_id])
        #update documents_df
        documents_df.loc[documents_df['id']==document_id, 'num_paragraphs']=num_paragraphs
    
    return documents_df


def detect_signatures(paragraphs_df, styles_df):

    paragraph_matches=paragraphs_df[paragraphs_df['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)]
    print("Number of matches: ", len(paragraph_matches))

    document_ids=paragraph_matches['document'].value_counts()

    #if document id appears only once, then that's the beginning of the signature block
    #get document ids that appear only once in the dataframe
    matched_once_ids=document_ids[document_ids==1].index
    for doc_id in matched_once_ids:

        #get paragraph
        paragraph=paragraph_matches[paragraph_matches['document']==doc_id]

        #verify in styles_df that the text is not of type "quote" or "footnote"
        styles=styles_df[(styles_df['document']==doc_id) & (styles_df['paragraph_id']==paragraph['paragraph_id'].iloc[0])]
        styles_matches=styles[   styles['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)
                              & (styles['text_type']!='quote') & (styles['text_type']!='footnote')]
        #check that none of the styles are of type "quote" or "footnote"
        if len(styles_matches)>0:
            #get current paragraph section
            paragraph_section=paragraph['section'].iloc[0]
            #get paragraphs after current paragraph
            paragraphs_after=paragraphs_df[  (paragraphs_df['document']==doc_id) 
                                        & (paragraphs_df['paragraph_id']>paragraph['paragraph_id'].iloc[0]) 
                                        & (paragraphs_df['section']==paragraph_section)]
            #set section of paragraphs after  to "Signature Block"
            paragraphs_df.loc[paragraphs_after.index, 'section']="Signature Block"
            #get styles for paragraphs after
            styles_after=styles_df[(styles_df['document']==doc_id) & (styles_df['paragraph_id'].isin(paragraphs_after['paragraph_id']))]
            #set section of styles after to "Signature Block"
            styles_df.loc[styles_after.index, 'section']="Signature Block"

    #if document id appears more than once, then we need to check the distance between the matches
    multi_matches_ids=document_ids[document_ids>1].index
    
    while (len(multi_matches_ids)>0):
        curr_doc_id=multi_matches_ids[0]
        #get paragraphs sorted by paragraph_id
        paragraphs=paragraph_matches[paragraph_matches['document']==curr_doc_id].sort_values(by=['paragraph_id'])
        
        ind=0

        curr_id=paragraphs['paragraph_id'].iloc[ind]

        while ind<=len(paragraphs)-2:
            
            next_id=paragraphs['paragraph_id'].iloc[ind+1]

            #verify in styles_df that the text is not of type "quote" or "footnote"
            styles=styles_df[(styles_df['document']==curr_doc_id) & (styles_df['paragraph_id']==curr_id)]
            styles_matches=styles[   styles['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)
                                & (styles['text_type']!='quote') & (styles['text_type']!='footnote')]
            #check that none of the styles are of type "quote" or "footnote"
            if len(styles_matches)>0:
                if next_id-curr_id>8:
                    assign_section(curr_doc_id, curr_id, "Signature Block", paragraphs_df, styles_df)   

            ind=ind+1
            curr_id=paragraphs['paragraph_id'].iloc[ind]
        
        #verify in styles_df that the text is not of type "quote" or "footnote"
        styles=styles_df[(styles_df['document']==curr_doc_id) & (styles_df['paragraph_id']==next_id)]
        styles_matches=styles[   styles['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)
                            & (styles['text_type']!='quote') & (styles['text_type']!='footnote')]
        #check that none of the styles are of type "quote" or "footnote"
        if len(styles_matches)>0:
            assign_section(curr_doc_id, next_id, "Signature Block", paragraphs_df, styles_df)
        
        #remove document id from list
        multi_matches_ids=multi_matches_ids[1:]

    return paragraphs_df, styles_df


def assign_section(document_id, starting_paragraph_id, new_section_name, paragraphs_df, styles_df):

    #get section name for paragraph
    section_name=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==starting_paragraph_id)]['section'].iloc[0]

    #get paragraphs after current paragraph
    paragraphs_after=paragraphs_df[  (paragraphs_df['document']==document_id) 
                                & (paragraphs_df['paragraph_id']>starting_paragraph_id) 
                                & (paragraphs_df['section']==section_name)]
    #set section of paragraphs after  to "Signature Block"
    paragraphs_df.loc[paragraphs_after.index, 'section']=new_section_name
    #get styles for paragraphs after
    styles_after=styles_df[(styles_df['document']==document_id) & (styles_df['paragraph_id'].isin(paragraphs_after['paragraph_id']))]
    #set section of styles after to "Signature Block"
    styles_df.loc[styles_after.index, 'section']="Signature Block"

    return paragraphs_df, styles_df


def get_document_id(filename, df_docs):

    #remove extension form filename
    extension_index=filename.rfind(".")
    if extension_index!=-1:
        filename=filename[:extension_index]
        #print("filename: ", filename)
    #find id for row that matches filename
    doc=df_docs.loc[df_docs["name"]==filename].iloc[0]
    return doc['id']

#TODO: Verify if get_decision_id is necessary and see if we can merge some functionality with get_document_id

def get_decision_id(filename, documents_df):
    #get document info
    name=filename.split("/")[-1]

    #find row in documents_df
    row=documents_df.loc[documents_df["name"]==name][0]
    #get document id
    doc_id=row["id"]
    return doc_id
