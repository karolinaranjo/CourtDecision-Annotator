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


def get_text_type(tag, tagless_text, usual_font_size):
    '''
    Determines text type based on the font size and other attributes.

    Parameters
    ----------
    tag :str
        The tag associated with the text
    tagless_text :str
        text without any tags.

    usual_font_size : int
        The usual font size of the text.

    Returns
    -------
    text_type : str
        The type of the text based on its attributes:'title', 'footnote', 
        'emphasis', 'quote', and 'plain'.

    '''
    #get font size
    fs=kpdf.get_trailing_number(tag)
    
    text_type=""
    text_type_es=""
    #use font size to establish type
    if fs>usual_font_size:
        text_type=add_info_to_string(text_type,"title")
        text_type_es=add_info_to_string(text_type_es,"título")

    elif tagless_text.startswith("Página No."):
        text_type=add_info_to_string(text_type,"page_number")
        text_type_es=add_info_to_string(text_type_es,"número de página")

    # elif fs<usual_font_size:
    #     text_type=add_info_to_string(text_type,"footnote")

    elif fs<=usual_font_size-4:
        text_type=add_info_to_string(text_type,"footnote")
        text_type_es=add_info_to_string(text_type_es,"pie de página")

    if "page_number" not in text_type:
    
        if tagless_text.isupper() and "title" not in text_type:
            text_type=add_info_to_string(text_type,"emphasis")
            text_type_es=add_info_to_string(text_type_es,"énfasis") 

        if "italic" in tag and "footnote_number" not in text_type:
            text_type=add_info_to_string(text_type,"quote")
            text_type_es=add_info_to_string(text_type_es,"cita")
        
        if "bold" in tag and "emphasis" not in text_type and "footnote_number" not in text_type:
            text_type=add_info_to_string(text_type,"emphasis")
            text_type_es=add_info_to_string(text_type_es,"énfasis")
   
    if text_type=="":
        text_type="plain"
        text_type_es="normal"

    return text_type, text_type_es


def add_info_to_string(old_str, param):
    '''
    Accepts an existing string and a parameter as inputs, and produces a new string 
    that is a combination of the existing string and the parameter, with an underscore separating them.

    Parameters
    ----------
    old_str : str
        Old string.
    param :str
        parameter to be added add the end of the old string.

    Returns
    -------
    new_str : str
        Updated string separated by an underscore.

    '''
    if len(old_str)==0:
        new_str=param
    else:
        new_str=old_str+"_"+param
    return new_str

def process_paragraph_text(text, doc_id, paragraph_id, paragraph_text, df, usual_font_size, paragraph_doc_pos=0):
    '''
    Processes paragraph text from documents and returns a dataframe with extracted information (XML tags)
    List of tuples containing the text type and text length.

    Parameters
    ----------
    text : str
        Paragraph text with inner tags (if any).
    doc_id : Int
        Document ID.
    paragraph_id : int
        Paragraph identifier.
    paragraph_text : Str
        Should be raw text (no tags).
    df : Pandas dataframe 
        A dataframe of paragraphs for a document collection.
    usual_font_size : int
        The usual font size of the text.
    paragraph_doc_pos : int, optional
        The position of the paragraph in the document. The default is 0.

    Returns
    -------
    df : pd dataframe
       updated paragraphs dataframe.
    par_xml : str
        An XML representation of the paragraph.
    par_info : List[Tuple[str, int]]
        A list of tuples containing the text type and text length.

    '''

    new_text=text

    first_tag=kpdf.get_first_opening_tag(new_text)
    
    par_xml=""
    par_info=[]
    cursor=0
    cursor_untagged=0

    if first_tag!="d'oh": #if there are tags to be processed
        
        end_of_paragraph=False
        while not end_of_paragraph:
            
            tagless_text, tagged_text, ini_tagless, ini_tagged=kpdf.get_text_between_tags2(new_text, first_tag)
            
            text_length=len(tagless_text)
            #cursor=cursor+ini_tagged+len(tagged_text)
            
            #get text type
            text_type, text_type_es = get_text_type(first_tag, tagless_text, usual_font_size)
            
            #add to list of text types
            par_info.append((text_type, text_length))
            
            #get the position within the paragraph
            paragraph_pos=cursor_untagged
            #paragraph_pos=new_text.find(tagless_text)+paragraph_doc_pos
            #paragraph_pos= paragraph_text.find(tagless_text)

            #TODO: fix paragraph pos so that it searches for the latest occurrence of the text
            doc_pos=paragraph_doc_pos + paragraph_pos
            
            #save to dataframe
            df.loc[len(df)] = [doc_id, paragraph_id, "Pending", text_type, text_type_es, text_length, doc_pos, paragraph_pos, tagless_text]

            #save to xml
            par_xml+="<"+text_type+">"+tagless_text+"</"+text_type+">"
            
            cursor=cursor+ini_tagged+len(tagged_text)
            cursor_untagged=cursor_untagged+len(tagless_text)

            new_text=text[cursor:]
            first_tag=kpdf.get_first_opening_tag(new_text)
            if first_tag=="d'oh": #no inner tag, we're done
                end_of_paragraph=True
    
    return df, par_xml, par_info

def get_semantic_blocks_from_XML(full_text, doc_id, paragraphs_df=None, styles_df=None):
    """
    Processes a document’s text and updates dataframes for paragraphs and excerpts. 

    Parameters
    ----------
    full_text : Str
        The full text of the document.
    doc_id : Int
        The document ID.
    paragraphs_df : Pandas dataframe, optional
        A DF containing information about the paragraphs in the document collection.
        The default is None.
    styles_df : Pandas dataframe optional
        A dataframe containing information about the excerpts in the document. 
        The default is None.

    Returns
    -------
    styles_df : pd.DataFrame
        The updated dataframe containing information about the excerpts
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
    
    if styles_df is None:
        print("Creating new styles dataframe...")
        
        #styles_df=pd.DataFrame(columns=['document', 'paragraph_id', 'section', 'text_type', 'length','document_pos', 'paragraph_pos', 'function','text'])
        styles_df=pd.DataFrame(columns=['document', 'paragraph_id', 'section', 'text_type', 'text_type_es', 'length', 'document_pos', 'paragraph_pos', 'text'])
        #name index column
        styles_df.index.name = 'id'

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
        
        #new_section=get_section(p, paragraph_text, current_section)
        new_section=current_section
        if new_section!=current_section:
            current_section=new_section
            sections.append((current_section, p))
        
        #save to dataframe
        styles_df, paragraph_xml, paragraph_info=process_paragraph_text(p, doc_id, i, paragraph_text, styles_df, usual_font_size, doc_pos)

        par_fun="Pending"
        if i>0:
            paragraphs_df.loc[len(paragraphs_df)-1, 'next'] = i

        paragraphs_df.loc[len(paragraphs_df)] = [doc_id, i, i-1, -1, len(paragraph_text), doc_pos, current_section, par_fun, paragraph_text.strip("\n")]
        
        doc_xml+="<paragraph>\n"+paragraph_xml+"\n</paragraph>\n"
        doc_pos=doc_pos+len(paragraph_text)
        
    doc_xml+="</document>"

    return styles_df, paragraphs_df, sections, doc_xml



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

    formats=[]
    previous_number=0
    for entry in res:

        print("entry id", entry['paragraph_id'])
        print("entry text", entry['text'])
        print("previous paragraph id", entry['previous'])
        piece=entry['text'].strip()[0:15]
        current_format, current_number=krusty.detect_numerals_by_format(piece)

        print("current_format", current_format)
        print("current_number", current_number)

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

                index_next=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['next'])].index[0]
                paragraphs_df.loc[index_next, 'previous']=previous_par['paragraph_id']
                #print("updated next paragraph", paragraphs_df.loc[index_next])

                ids=styles_df[(styles_df['document']==document_id) &(styles_df['paragraph_id']==entry['paragraph_id'])].index

                for id in ids:
                    styles_df.loc[id, 'paragraph_id']=previous_par['paragraph_id']
                    styles_df.loc[id, 'paragraph_pos']=len(previous_par['text'])+styles_df.loc[id, 'paragraph_pos']
                
                index_current=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['paragraph_id'])].index[0]
                paragraphs_df.drop(index_current, inplace=True)

                current_format=previous_par_format
                
        formats.append(current_format)
        
    print("Formats before pruning:", formats)
    
    formats = [x for x in formats if x != "unknown"]
    print("Formats after pruning:", formats)
    most_common_format=krusty.most_common_item(formats)

    if most_common_format is not None:
        if most_common_format=="ara-hyphen":
            pattern = r"^\b\d+\b[ ]*[-][^0-9]"
        elif most_common_format=="ara-dot":
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
            aux_text=p['text'].strip()[0:15]
            aux=re.search(numbering_pattern, aux_text)
            if aux and len(aux_text.split())>0:
                print("match", aux.group())
                print("paragraph text", p['text'])
                next_up=krusty.next_number(aux_text.split()[0])
                if next_up>=current_number:
                    matching_paragraphs.append(p)

                    current_number=next_up
        
    for row in matching_paragraphs:
        print("document:", row["document"], "paragraph_id:", row["paragraph_id"], "text:", row['text'], sep=", ")
        
    #sort matching paragraphs by paragraph_id
    matching_paragraphs.sort(key=lambda x: x['paragraph_id'])
        
    return matching_paragraphs

def update_document_sections(document_id, paragraphs_df, styles_df, sections, case=False):
    '''
    Updates the section names for the paragraphs and excerpts in the dataframes and returns the updated dataframes.

    Parameters
    ----------
    document_id : int
        The ID of the document.
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
    sections : List
        A list containing section names and corresponding paragraph IDs.

    Returns
    -------
    paragraphs_df : pd.DataFrame
        The updated dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        The updated dataframe containing information about the excerpts.
    '''

    section_list, section_names_re=get_section_mapping()

    i=0
    while i<len(sections):
        
        section_id=sections[i]['paragraph_id']
        section_name=match_sections_to_names(sections[i]['text'][0:50], case) #only use first 50 characters to match section name, experiment with this
        if i==len(sections)-1:
            next_section_id=float('inf')
        else:
            next_section_id=sections[i+1]['paragraph_id']

        indices=paragraphs_df[(paragraphs_df['paragraph_id']>=section_id) & (paragraphs_df['paragraph_id']<next_section_id) & (paragraphs_df['document']==document_id)].index

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

        #update the section name for the styles/excerpts
        indices=styles_df[(styles_df['document']==document_id) & (styles_df['paragraph_id']>=section_id) & (styles_df['paragraph_id']<next_section_id)].index
        styles_df.loc[indices, 'section']=section_name

        i+=1    

    print("Section names have been updated")
    return paragraphs_df, styles_df

def match_sections_to_names(text, case=False):
    '''
    Takes in a text string and if the text corresponds to a section name,
    it returns the standard name of the section that matches the text.

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
        A dataframe containing information about the excerpts.
    documents_df : pd.DataFrame
        A dataframe containing information about the documents.

    Returns
    -------
    paragraphs_df : pd.DataFrame
        The updated dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        The updated dataframe containing information about the excertps.
    documents_df : pd.DataFrame
        The updated dataframe containing information about the documents.

    '''

    if 'num_sections' not in documents_df.columns:
        documents_df['num_sections']=0

    documents=paragraphs_df["document"].unique()

    section_names=set()

    for document in documents:
        document_name=documents_df[documents_df['id']==document]['name'].iloc[0]
        print("Updating sections for document ", document, "(", document_name, ")")
        common_format, pattern=find_section_format(paragraphs_df, styles_df, document)
        if common_format!="unknown":
            sections=find_sections(paragraphs_df, common_format, pattern, document)

            new_section_names=get_section_names(sections)

            section_names=section_names.union(new_section_names)

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
    Compiles text type leghts by decision section for a specified document,using paragraphs and excerpts data.
    
    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
    document_id : int
        The ID of the document.

    Returns
    -------
    res_df : pd.DataFrame
        Dataframe with summary of text lengths  by text type and section.
    '''
    sections=paragraphs_df[paragraphs_df['document']==document_id]['section'].unique()
    for section in sections:
        section_df=styles_df[(styles_df['document']==document_id) & (styles_df['section']==section)]
        text_types=section_df['text_type'].unique()
        for text_type in text_types:
            length=section_df[section_df['text_type']==text_type]['length'].sum()
            print("Document:", document_id, "Section:", section, "Text type:", text_type, "Length:", length) if verbose==True else ""

    res_df=styles_df.groupby(['document','section', 'text_type']).agg({'length': 'sum'}).reset_index()
    
    return res_df


def get_sections_statistics(paragraphs_df, styles_df, verbose=False):
    '''
    Aggregates lengths of text types by section accross all document collection using paragraphs and excerpts data. 

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
    verbose: bool
        Enables detailed output. Default is False

    Returns
    -------
    res_df : pd.DataFrame
        A dataframe with statistics of text type lengths for each document section.
    '''
    
    documents=paragraphs_df['document'].unique()
    res_df=pd.DataFrame(columns=['document', 'section', 'text_type', 'length'])
    for document in documents:
        cur_res_df=get_document_statistics(paragraphs_df, styles_df, document, verbose)
        res_df=pd.concat([res_df, cur_res_df], ignore_index=True)
        
    return res_df


def time_analysis(documents_df, stats_df, type="X"):
    '''
    Computes the average lengths of sections and text types within documents for each year, categorizing by document type when specified.

    Parameters
    ----------
    documents_df : pd.DataFrame
       Contains document information.
    stats_df : pd.DataFrame
       Holds statistics on sections and text types.
    type : str, optional
       Filters analysis to a specific document type, with "X" indicating all types. Default is "X".

    Returns
    -------
    section_res_df : pd.DataFrame
        Average section lengths by year.
    sec_text_types_res_df : pd.DataFrame
        Average text type lengths by section and year.
    '''
    if type!="X":
        docs_df=documents_df[documents_df['type']==type].copy()
        print(len(docs_df), " documents of type ", type, " found")
    else:
        docs_df=documents_df.copy()
        print("Analyzing all ". len(docs_df), " documents...")

    text_types=stats_df['text_type'].unique()
    years=docs_df['year'].unique()
    sections=stats_df['section'].unique()
    years.sort()
    #store results
    section_res_df=pd.DataFrame(columns=['year', 'section', 'avg_length'])
    sec_text_types_res_df=pd.DataFrame(columns=['year', 'section', 'text_type', 'avg_length'])

    for year in years:

        doc_ids_year=docs_df[docs_df['year']==year]['id'].unique()

        for section in sections:
            if section!="Pending":
                #get average length for the year and section
                avg_length=stats_df[(stats_df['document'].isin(doc_ids_year)) & (stats_df['section']==section)]['length'].mean()
                if np.isnan(avg_length):
                    avg_length=0
                print("Year:", year, "Section:", section, "Average length:", avg_length)

                section_res_df.loc[len(section_res_df)] = [year, section, avg_length]
                #get average length for each text type for the year and section
                for text_type in text_types:
                    avg_length=stats_df[(stats_df['document'].isin(doc_ids_year)) & (stats_df['section']==section) & (stats_df['text_type']==text_type)]['length'].mean()
                    if np.isnan(avg_length):
                        avg_length=0
                    print("Year:", year, "Section:", section, "Text type:", text_type, "Average length:", avg_length)
                    #save to dataframe
                    sec_text_types_res_df.loc[len(sec_text_types_res_df)] = [year, section, text_type, avg_length]

    return section_res_df, sec_text_types_res_df

def detect_verbs(text):
    '''
    Analyzes a given text to identify and return a list of verbs using the SpaCy lib

    Parameters
    ----------
    text : str
        The input text to analyze for verb detection.

    Returns
    -------
    verbs : str
        A list containing the verbs found in the input text.
    '''
    verbs = []
    nlp = spacy.load("es_core_news_sm")
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
        A dataframe containing information about the excerpts.
    verbose : bool, optional
        If True, the function will print additional information.
        The default is False.

    Returns
    -------
    verbs : List[str]
        A list of verbs found in the excerpts.
    positive_words : List[str]
        A list of positive words found in the excerpts.
    negative_words : List[str]
        A list of negative words found in the excerpts.

    '''
    
    verbs=[]
    positive_words=[]
    negative_words=[]

    matched_blocks=blocks_df[blocks_df['text_type']==text_type]

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


def get_file_metadata_from_db(filename, conn=None):
    """
    Retrieves metadata for a specified file from a database, using an existing database connection.
    
    Parameters
    ----------
    filename : str
        Name of the file to search in the database.
    conn : sqlite3.Connection, optional
        Existing SQLite database connection.The default is None.

    Returns
    -------
    r : tuple
        Contains the first matching record.
    conn : conn
        The database connection.

    """
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
    """
    Classifies paragraphs and excerpts related to dissenting opinions within a decision document.
    Updates sections in the provided DataFrames accordingly.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.

    Returns
    -------
    paragraphs_df : pd DataFrame
        Updated dataframe with new paragraphs for dissenting opinions
    styles_df : pd DataFrame
        Updated dataframe with the correct section assignment for dissenting opinions

    """

    paragraphs_df=paragraphs_df.copy()
    styles_df=styles_df.copy()

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
        paragraph_index=paragraph.index[0]

        if len(paragraph['text'].iloc[0])<len(row['text'])+100:
            #change section to "Salvamento de voto"
            paragraphs_df.loc[paragraph_index, 'section']="Dissenting Opinion - " + str(current_opinion)
            current_opinion=current_opinion+1

            for ind_block, block in styles_df[(styles_df['document']==row['document']) & (styles_df['paragraph_id']==row['paragraph_id'])].iterrows():
                styles_df.loc[ind_block, 'section']="Dissenting Opinion - " + str(index)

    paragraphs_df['opinion'] = paragraphs_df.section.str.extract('(\d)')
    paragraphs_df['aux_section'] = paragraphs_df['section'].copy()

    sections=paragraphs_df['section'].unique()

    sections=pd.Series(sections)
    #get all sections that do not contain the text "Dissenting Opinion"
    sections=sections[~sections.str.contains("Dissenting Opinion")]
    for s in sections:
        paragraphs_df['section']=paragraphs_df['section'].replace(s, np.nan)

    paragraphs_df=paragraphs_df.groupby(by='document', group_keys=False
                        ).apply(fill_opinion         # propagate dissenting opinions
                        ).reset_index(drop=True      # clean-up index
                        ).drop(columns = 'opinion')  #remove temporary column

    paragraphs_df['section'] = np.where(paragraphs_df['section'].isna(), paragraphs_df['aux_section'], paragraphs_df['section'])
    #drop aux_section column
    paragraphs_df=paragraphs_df.drop(columns=['aux_section'])
    
    #drop section column
    styles_df=styles_df.drop(columns=['section'])
    #assign paragraph's section to styles that match document and paragraph_id
    styles_df=styles_df.merge(paragraphs_df[['document', 'paragraph_id', 'section']], on=['document', 'paragraph_id'], how='left')

    return paragraphs_df, styles_df


def detect_signatures(paragraphs_df, styles_df):
    """
    Updates sections in documents to "Signature Block" based on the presence of signature-related keywords.

    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
        
    Returns
    -------
    paragraphs_df : pd DataFrame
        Updated dataframe with "signature block" section.
    styles_df : pd DataFrame
        Updated dataframe with "signature block" section.
    """
    
    paragraph_matches = paragraphs_df[paragraphs_df['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)]
    print("Number of matches: ", len(paragraph_matches))

    document_ids=paragraph_matches['document'].value_counts()

    #if document id appears only once, then that's the beginning of the signature block
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

            styles=styles_df[(styles_df['document']==curr_doc_id) & (styles_df['paragraph_id']==curr_id)]
            styles_matches=styles[   styles['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)
                                & (styles['text_type']!='quote') & (styles['text_type']!='footnote')]
            #check type "quote" or "footnote"
            if len(styles_matches)>0:
                if next_id-curr_id>8:
                    assign_section(curr_doc_id, curr_id, "Signature Block", paragraphs_df, styles_df)   

            ind=ind+1
            curr_id=paragraphs['paragraph_id'].iloc[ind]
        
        #verify that its not of type "quote" or "footnote"
        styles=styles_df[(styles_df['document']==curr_doc_id) & (styles_df['paragraph_id']==next_id)]
        styles_matches=styles[   styles['text'].str.contains("notifíquese|cúmplase|publíquese|líbrese|notifiquese|líbrense|cumplase", case=False)
                            & (styles['text_type']!='quote') & (styles['text_type']!='footnote')]
        
        if len(styles_matches)>0:
            assign_section(curr_doc_id, next_id, "Signature Block", paragraphs_df, styles_df)
        #remove document id 
        multi_matches_ids=multi_matches_ids[1:]

    return paragraphs_df, styles_df

def fill_opinion(df_d):
    
    '''
    Fills "dissenting opinions" for one document
    '''
    
    return df_d.assign(section = 'Dissenting Opinion - ' + df_d['opinion']).ffill()

def fill_missing_section(row):
    """
    Returns aux_section if present or section field otherwise.

    Parameters
    ----------
    row : DataFrame row
        Row to be modified

    Returns
    -------
    str
        The 'section' value if present, otherwise the 'aux_section' value.

    """
    if pd.isna(row['section']):
        return row['aux_section']
    else:
        return row['section']

def get_section_names(section_rows):
    """
    Extracts section names from a dataframe containing beginning-of-section rows

    Parameters
    ----------
    section_rows : List[Dict]
        Contains dictionary-like objects with a 'text' key.

    Returns
    -------
    section_names : List[str]
        List of section names found in the input dataframe.

    """
    section_names=[]
    for row in section_rows:
        section_names.append(row['text'])
    return section_names

def check_section_integrity(paragraphs_df, styles_df, document_id):
    """
    Checks section rows to verify the consistency of numbering and naming patterns.
    It also fixes some errors such as section names that had been split into 
    two different paragraphs.
    
    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
    document_id : int
        Identifier for the target document.

    """
    section_rows=paragraphs_df[paragraphs_df['document']==document_id].groupby('section').first().reset_index().sort_values(by=['paragraph_id'])
    
    display(section_rows) #testing purposes

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

    smallest_number=min(section_numbers)
    largest_number=max(section_numbers)

    missing_sections=[]
    for i in range(smallest_number, largest_number+1):
        if i not in section_numbers:
            print("Section number ", i, " is missing")
            missing_sections.append(i)

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

            for index, row in missing_sections_df.iterrows():
                print("Text: ", row['text'])
                aux_row_text=row['text'].strip().replace(".", "").replace("-", "").replace(" ", "").replace(number, "")
                if len(aux_row_text)==0: #if the text is only the number, then we should merge it with the next paragraph
                    print("Merging with next paragraph")
                    entry=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==row['next'])].iloc[0]

                    previous_par=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['previous'])].iloc[0]
                    previous_par=row

                    index=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==previous_par['paragraph_id'])].index[0] 
                    paragraphs_df.loc[index, 'text']=previous_par['text']+entry['text']

                    #update 'next' of previous paragraph
                    paragraphs_df.loc[index, 'next']=entry['next']

                    #update 'previous' of new next paragraph
                    index_next=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['next'])].index[0]
                    paragraphs_df.loc[index_next, 'previous']=previous_par['paragraph_id']

                    ids=styles_df[(styles_df['document']==document_id) &(styles_df['paragraph_id']==entry['paragraph_id'])].index

                    for id in ids:
                        styles_df.loc[id, 'paragraph_id']=previous_par['paragraph_id']
                        styles_df.loc[id, 'paragraph_pos']=len(previous_par['text'])+styles_df.loc[id, 'paragraph_pos']

                    index_current=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==entry['paragraph_id'])].index[0]
                    paragraphs_df.drop(index_current, inplace=True)
    else:
        print("No missing sections")

def check_section_integrity_by_document(paragraphs_df, styles_df):
    """
    Assesses section integrity within each document in provided DataFrames.
    
    Parameters
    ----------
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
    """
    documents=paragraphs_df['document'].unique()
    for d in documents:
        print("Checking document: ", d)
        check_section_integrity(paragraphs_df, styles_df, d)

def get_sections_without_number(document_id, paragraphs_df, styles_df):
    """
    Finds paragraphs matching standard section names without numerals and of text type "emphasis".

    
    Parameters
    ----------
    document_id : int
        Target document ID.
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.
        
    Returns
    -------
    res : list of dict
        A list of rows, each representing a paragraph that is a section name without a numeral.
    """
    _, section_names_re=get_section_mapping()
    #convert to uppercase
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
                styles=styles_df[(styles_df['document']==matches.iloc[0]['document']) & (styles_df['paragraph_id']==matches.iloc[0]['paragraph_id'])]
                #check if all styles are of type "emphasis"
                if len(styles[styles['text_type']!='emphasis'])==0:
                        res.append(matches.iloc[0])

    return res

def get_section_mapping():
    """
    Returns lists of common decision document section names and their aliases. 

    Returns
    -------
    sections : list of str
        Standard section names of judicial decisions.
    section_names_re : list of lists
        Typical aliases for each section name. Section search is performed
        against this list.

    """

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

def fuzzy_section_mapping(df):
    """
    Identifies the closest matches for "pending" sections using predefined mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Contains sections marked as "pending" for analysis.
    """
    sections, section_names_re=get_section_mapping()

    pending_df=df[df['section'].str.contains("pending", case=False) & (df['section'].shift(1)!=df['section'])]

    for index, row in pending_df.iterrows():
        section, matched_term=find_closest_match(sections, section_names_re, row['text'])
        
        print("Text: ", row['text'])
        print("Matched term: ", section)
        print("Matched terms: ", matched_term)

def find_closest_match(terms, associated_vocab, word):
    """
    Returns a term for a word by finding the closest vocab match in the 
    list of associated vocab. It uses the Levenshtein distance to find the 
    vocab match.
    
    Parameters
    ----------
    terms :List of str
        List of possible terms.
    associated_vocab :List
        The vocabulary associated with each of the terms.
    word : str
        Word to find the matching term.
        
    Returns
    -------
    match : str
        The term that most closely matches the given word.
    associated_vocab[min_index]: List of str
        The vocabulary associated with the closest matching term.
    min_distance : int
        The minimum Levenshtein distance between the word and the matched term's vocabulary.
    """

    i=0
    term_distances=[]
    while i <len(terms):
        associated_distances=[]
        j=0
        while j<len(associated_vocab[i]):
            associated_distances.append(Levenshtein.distance(word, associated_vocab[i][j]))
            j=j+1
        min_distance=min(associated_distances)
        term_distances.append(min_distance)
        i=i+1
    min_distance=min(term_distances)
    min_index=term_distances.index(min_distance)
    match=terms[min_index]
    
    return match, associated_vocab[min_index], min_distance

def set_number_of_paragraphs_per_document(documents_df, paragraphs_df):
    """
    Updates documents_df by including paragraphs counts. 

    Parameters
    ----------
    documents_df :pd.DataFrame
        A dataframe containing document metadata.
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    Returns
    -------
    documents_df : pd.DataFrame
        Enhanced dataframe with a num_paragraphs column indicating the count of paragraphs per document.

    """
    if 'num_paragraphs' not in documents_df.columns:
        documents_df['num_paragraphs']=0
    document_ids=documents_df['id'].unique()

    for document_id in document_ids:
        num_paragraphs=len(paragraphs_df[paragraphs_df['document']==document_id])
        documents_df.loc[documents_df['id']==document_id, 'num_paragraphs']=num_paragraphs
    
    return documents_df

def assign_section(document_id, starting_paragraph_id, new_section_name, paragraphs_df, styles_df):
    """
    Designates a new section name to paragraphs and their excerpts starting from a specified paragraph within a document.

    Parameters
    ----------
    document_id : int
        Document ID.
    starting_paragraph_id : int
        The ID of the paragraph where the new section begins.
    new_section_name : str
        The name of the new section.
    paragraphs_df : pd.DataFrame
        A dataframe containing information about the paragraphs.
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.

    Returns
    -------
    paragraphs_df :pd DataFrame
        Updated dataframe with paragraphs after the starting paragraph set to the new section name.
    styles_df : pd DataFrame
        Updated dataframe with paragraphs after the starting paragraph set to the new section name.
    """
    section_name=paragraphs_df[(paragraphs_df['document']==document_id) & (paragraphs_df['paragraph_id']==starting_paragraph_id)]['section'].iloc[0]

    paragraphs_after=paragraphs_df[  (paragraphs_df['document']==document_id) 
                                & (paragraphs_df['paragraph_id']>starting_paragraph_id) 
                                & (paragraphs_df['section']==section_name)]

    paragraphs_df.loc[paragraphs_after.index, 'section']=new_section_name

    styles_after=styles_df[(styles_df['document']==document_id) & (styles_df['paragraph_id'].isin(paragraphs_after['paragraph_id']))]

    styles_df.loc[styles_after.index, 'section']="Signature Block"

    return paragraphs_df, styles_df


def get_document_id(filename, df_docs):
    """
    Retrieves the document ID associated with a given filename from the docs_df.

    Parameters
    ----------
    filename : str
        The name of the file
    df_docs : pd.DataFrame
        Contains document information (names and IDs).
    Returns
    -------
    id: int
        ID information corresponding to the provided filename, after removing its extension.

    """
    #remove extension form filename
    extension_index=filename.rfind(".")
    if extension_index!=-1:
        filename=filename[:extension_index]
 
    doc=df_docs.loc[df_docs["name"]==filename].iloc[0]
    return doc['id']

def get_decision_id(filename, documents_df):
    """
    Extracts the decision ID associated with a specific filename from documents_df.

    Parameters
    ----------
    filename : str
        The full name of the file.
    documents_df :pd.DataFrame
        Contains document metadata.

    Returns
    -------
    doc_id : int
        The ID of the document that matches the given filename.
    """
    name=filename.split("/")[-1]
    #find row in documents_df
    row=documents_df.loc[documents_df["name"]==name][0]
    doc_id=row["id"]
    return doc_id

def footnote_remapping(styles_df):
    """
    Migrates footnotes from v1 to v2 to make them more readable.

    Parameters
    ----------
    styles_df : pd.DataFrame
        A dataframe containing information about the excerpts.

    Returns
    -------
    styles_df : pd.DataFrame
        Updated dataframe with remapped footnotes.
    
    """

    #get all excerpts of type "footnote"
    #footnotes=styles_df[styles_df['text_type'].str.startswith('footnote')]
    footnotes=styles_df[styles_df['text_type'].str.contains('footnote')]

    for index, row in footnotes.iterrows():
        if row['paragraph_pos']>5 and row['length']<10: #it must be a footnote number 
            styles_df.loc[index, 'text_type']="footnote_number"
            styles_df.loc[index, 'text_type_es']="número de pie de página"

    #find rows in which text_type is footnote_something (but not footnote_number)
    #footnotes=styles_df[ (styles_df['text_type']!='footnote_number') & (styles_df['text_type'].str.startswith('footnote')) ]
    footnotes=styles_df[ (styles_df['text_type']!='footnote_number') & (styles_df['text_type'].str.contains('footnote')) ]

    for index, row in footnotes.iterrows():
        #find previous row that belongs to the same paragraph using paragraph position
        aux_previous=styles_df[(styles_df['document']==row['document']) & (styles_df['paragraph_id']==row['paragraph_id']) & (styles_df['paragraph_pos']<row['paragraph_pos'])]
        if len(aux_previous)>0:
            #get previous block
            previous_block=aux_previous.loc[aux_previous['paragraph_pos'].idxmax()]
            if previous_block['text_type']=="page_number":
                #remove footnote from text type in the current row and all following rows that belong to the same paragraph
                styles_ahead_df=styles_df[(styles_df['document']==row['document']) & (styles_df['paragraph_id']==row['paragraph_id']) & (styles_df['paragraph_pos']>=row['paragraph_pos'])]
                for ind, r in styles_ahead_df.iterrows():
                    styles_df.loc[ind, 'text_type']=r['text_type'].replace("footnote_", "").strip("_")
                    styles_df.loc[ind, 'text_type_es']=r['text_type_es'].replace("pie de página ", "").strip(" ")
               

    for index, row in footnotes.iterrows():
        #find rows that belong to the same paragraph and document
        same_paragraph=styles_df[(styles_df['document']==row['document']) & (styles_df['paragraph_id']==row['paragraph_id'])]
        for ind, par in same_paragraph.iterrows():
            #if text_type doesn't contain "footnote"
            if "footnote" not in par['text_type']:
                if par['text_type']=="plain":
                    styles_df.loc[ind, 'text_type']="footnote"
                    styles_df.loc[ind, 'text_type_es']="pie de página"
                else:
                    styles_df.loc[ind, 'text_type']="footnote_"+par['text_type']
                    styles_df.loc[ind, 'text_type_es']="pie de página "+par['text_type_es']


    return styles_df
