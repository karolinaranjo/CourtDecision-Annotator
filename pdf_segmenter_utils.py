#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 08:19:49 2023

@author: KarolinaNaranjo
"""
import pandas as pd
import pickle
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar
from pdfminer.high_level import extract_pages, extract_text
import re
import kutils as krusty
import statistics
import os
from os.path import isfile, join
from pdf2image import convert_from_path

import pdf_segmenter_utils as kpdf
import semantic_annotations_utils as sau

font_fields=["name", "size", "label"]
font_field_types=["TEXT", "REAL", "TEXT"]
fonts_db="fonts_db.db"
fonts_table="fonts"

def clean_up_xml(text, verbose=False):
    '''
    Cleans up text in an XML-format by removing empty tags and labels that only contain spaces. 

    Parameters
    ----------
    text :String
        Contains the text to be cleaned up.
    verbose : Bool, optional
        If set to True, prints the labels that are being processed.
        The default is False.

    Returns
    -------
    text : String
        The cleaned XML text as a string.

    '''
    # remove empty paragraphs
    text=text.replace("<paragraph>\n\n</paragraph>", "")

    #pattern=re.compile(r'<[a-z0-9_]+>[ ]*<\/[a-z0-9_]+>') #labels but only spaces in between
    pattern=re.compile(r'<[\w]+>[ ]*<\/[\w]+>') #labels but only spaces in between
    #\w
    matches=pattern.finditer(text)
    for match in matches:
        text=text.replace(match[0], "")

    #join content if adjacent tags are the same
    pattern=re.compile(r'<[\w]+>') #find set of labels
    matches=pattern.finditer(text)
    for match in matches:
        label=match[0][1:-1]
        
        #print("label is "+label)
        print("label is "+label) if verbose==True else ""
        # check to see if two consecutive labels are not paragraphs. I stopped this
        # because I figured that this would only happen if the specifity of the labels
        # wasn't enough to detect an actual change: regular 14 and regular 14 might be
        # referring to two different font types!
    
    return text

def convert_pdf_files_to_text(input_folder, output_folder):
    '''
    Processes all PDF files in the determined input folder and extracts their text content into new text files in the output folder.

    Parameters
    ----------
    input_folder :String
        The folder contains the PDF files to be processed.
    output_folder :String
        The folder where the new text files will be saved.

    Returns
    -------
    str
        An empty string.

    '''
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")
    py_output_folder=output_folder.replace(r"\~", "~").replace(r"\ ", " ")
    
    files=os.listdir(py_input_folder)
    for f in files:
        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):
            output_file_path=join(py_output_folder, f.replace(".pdf", ".txt"))
            if isfile(output_file_path)==False:
                data = extract_text(file_path)
                #save new file
                f = open(output_file_path, "w")
                f.write(data)
                f.close()
            else: 
                print("There is already a file with the same name in the destination folder.")
        
    return ""
    
def convert_xml_files_to_raw_text(input_folder, output_folder):
    '''
    Converts XML files to raw text files and saves them in the output folder.
    If a file with same name already exists in the output folder, it won't be overwritten

    Parameters
    ----------
    input_folder : String
        The folder contains the XML files to be converted.
    output_folder : String
        The folder where the resulting raw text files will be saved.

    Returns
    -------
    None.

    '''
    
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")
    py_output_folder=output_folder.replace(r"\~", "~").replace(r"\ ", " ")
    
    files=os.listdir(py_input_folder)
    
    for f in files:
        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".xml"):
            output_file_path=join(py_output_folder, f.replace(".xml", ".txt"))
            if isfile(output_file_path)==False:
                with open(file_path, 'r') as file:
                    data = file.read()
                xml_to_raw_text_file(data, output_file_path)
            else: 
                print("There is already a file with the same name in the destination folder.")

                
def detect_paragraph_anomalies(paragraphs, text, verbose=False):

    '''
    Detects anomalies in the input paragraphs, such as lowercase letters or a font size that does not match the most common font size in the text, and joins them if necessary.
    
    Parameters
    ----------
    paragraphs : String
        A list of paragraphs to be processed
    text : String
        The text containing the paragraphs.
    verbose : bool, optional
        Controls whether or not additional infomation about the detected anomalies is printed.
        The default is False.

    Returns
    -------
    new_paragraphs : String
        A list of paragraphs with the detectr anomalies joined if necessary.
    '''
    
    #sanitized_text=remove_tags(text, ["paragraph", "page", "document"])
    sanitized_text=remove_tags(text, ["page", "document"])
    _, text_tags=get_tags(sanitized_text)
    #print(text_tags)
    text_size=get_most_common_font_size(text_tags)
    #print("text size is "+str(text_size))
    new_paragraphs=[]

    i=0
    num_par=len(paragraphs)
    while(i<num_par):
        
        _, p_tags=get_tags(paragraphs[i])
        aux_p=remove_tags(paragraphs[i], p_tags).strip()
        #print(aux_p)
        if aux_p[0].isalpha() and aux_p[0].islower() and get_paragraph_font_size(paragraphs[i])==text_size: #if paragraph starts with lowercase letter
            print("Anomaly detected:") if verbose==True else ""
            print("Paragraph #"+str(i)) if verbose==True else ""
            print(paragraphs[i]) if verbose==True else ""
            pf_size=get_paragraph_font_size(paragraphs[i])
            #is it a stray word?
            if len(aux_p.strip().split(" "))!=1:#if not a stray word
                print("It's not a stray word") if verbose==True else ""
                #look for the paragraph it should connect to
                j=len(new_paragraphs)-1
                found=False
                while found==False and j>=0:
                    if pf_size==get_paragraph_font_size(new_paragraphs[j]): #paragraphs that are joined must have the same font size to match
                        found=True
                        end_tag=get_last_closing_tag(new_paragraphs[j])
                        ini_tag=get_first_opening_tag(paragraphs[i])
                        
                        if end_tag==ini_tag:
                            end_pos=new_paragraphs[j].rfind("</"+end_tag+">")
                            ini_pos=paragraphs[i].find("<"+ini_tag+">")+len("<"+ini_tag+">")
                            new_paragraphs[j]=(new_paragraphs[j][0:end_pos]+" "+paragraphs[i][ini_pos:]).replace("  ", " ")
                        else:
                            new_paragraphs[j]=(new_paragraphs[j]+" "+paragraphs[i]).replace("  ", " ")
                    j=j-1
            else:
                print("Stray word!") if verbose==True else ""
                #new_paragraphs.append(paragraphs[i])#comment this line if you want to remove them
                
        else:
            new_paragraphs.append(paragraphs[i])
        i=i+1
        
    return new_paragraphs

                
def convert_pdf_files_to_sanitized_xml(input_folder, output_folder, font_db_conn):
    '''
    Converts PDF files to sanitized XML-format files and saves them in the output folder.
    If a file with same name already exists in the output folder, it won't be overwritten

    Parameters
    ----------
    input_folder :String
        The folder contains the PDF files to be converted.
    output_folder : String
        The folder where the resulting sanitized XML files will be saved.
     font_db_conn : object
         A connection to a font database to be used to find the label for each font type and size.

    Returns
    -------
    str
        A string indicating that the PDF processing is done.

    '''
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")
    py_output_folder=output_folder.replace(r"\~", "~").replace(r"\ ", " ")
    
    files=os.listdir(py_input_folder)
    for f in files:
        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):
            output_file_path=join(py_output_folder, f.replace(".pdf", ".xml"))
            if isfile(output_file_path)==False:
                #pdf_to_labeled_text(file_path, output_file_path, font_db_conn)
                xml=sanitized_xml_from_pdf(file_path, font_db_conn)
                
                output_f = open(output_file_path, "w")
                output_f.write(r""+xml)
                output_f.close()
                
            else: 
                print("There is already a file with the same name in the destination folder.")
        
    return "PDF processing done."


def find_font_label(conn, font_name, font_size, verbose=False):
    '''
    Finds the font label that matches the input font name and size in a database or generates a new one if not found.

    Parameters
    ----------
    conn :Object
        Connects to a database containing font information.
    font_name : String
        The name's font to find a label for.
    font_size : Float
        The font's size to find a label for.
    verbose : bool, optional
        Controls whether or not additional information about the found font label is printed.
        The default is False.

    Returns
    -------
    String
        Returns the found or generated font label.

    '''
    c=conn.cursor()
    #c.execute( "SELECT label FROM fonts WHERE name='"+font_name+"' AND size="+str(round(font_size,1)))
    c.execute( "SELECT label FROM fonts WHERE name='"+font_name+"' AND size="+str(round(font_size,1)))
    r=c.fetchone()
    #print(r[0])
    if r is None:
        if font_name.upper().find("BOLD")!=-1 and font_name.upper().find("ITALIC")!=-1:
            label="bold_italic"
        elif font_name.upper().find("BOLD")!=-1:
            label="bold"
        elif font_name.upper().find("ITALIC")!=-1:
            label="italic"
        else:
            label="regular"
        label=label+str(round(font_size))
        krusty.add_db_entry(conn, fonts_table, font_fields, [font_name, round(font_size,1), label])
        print('New font type added to database: '+font_name)
        return label
    else:
        print(r[0]) if verbose==True else ""
        return r[0]                
                
def pdf_to_labeled_text(file_name, font_db_conn, maxpages=-1, font_info=False):
    
    '''
    Extracts text content from a PDF file and returns it in an XML-format.
    
    Parameters
    ----------
    file_name : string
        It's the name of the PDF file to be processed.
    font_db_conn : object
        A connection to a font database to be used to find the label for each font type and size.
    maxpages : int, optional
        The maximum number of pages to process from the PDF file.
        The default is -1, meaning all pages will be processed.
    font_info : bool, optional
        It determines whether the output will include font tags with the font name and size (TRUE) or label tags based on the font database (FALSE)
        The default is False.

    Returns
    -------
    output_str : str
        It's a string that holds the text from the PDF file in an XML-format, organized into pages,
        paragraphs, and font or label tags.
    '''
    
    if maxpages==-1:
        pages=extract_pages(file_name)
    else:
        pages=extract_pages(file_name,maxpages=maxpages)
    
    output_str="<document>\n"
    
    curr_font=""
    curr_size=-1
    curr_label=""
    
    font_closing_tag='</font>'
    label_closing_tag=''
    tag_closed=True
    
    #document_fonts=[]
    
    for page_layout in pages:
        output_str=output_str+"<page>\n"

        for element in page_layout:
            if isinstance(element,LTTextContainer):
                output_str=output_str+"<paragraph>\n"
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        output_str=output_str+" "
                        for character in text_line:
                            if isinstance(character, LTChar):
                                if character.fontname==curr_font and character.size==curr_size: #if it's still same type
                                    output_str=output_str+character.get_text()
                                else: #if font type changed
                                    if curr_font!="": #if it's not the beginning of the file, close previous tag
                                        tag_closed=True
                                        if font_info==True:
                                            output_str=output_str+font_closing_tag
                                        else:
                                            output_str=output_str+label_closing_tag
                                        
                                    # get new properties
                                    curr_font=character.fontname
                                    curr_size=character.size
                                    curr_label=find_font_label(font_db_conn, curr_font, curr_size)
                                    if font_info==True:
                                        output_str=output_str+r'<font='+curr_font+str(round(curr_size))+'>'+character.get_text()
                                    else:
                                        output_str=output_str+r'<'+curr_label+'>'+character.get_text()
                                        label_closing_tag='</'+curr_label+'>'
                                    tag_closed=False
                if tag_closed==False:
                    if font_info==True:
                        output_str=output_str+font_closing_tag
                    else:
                        output_str=output_str+label_closing_tag

                    curr_font="" #reset tags to start new paragraph
                    tag_closed=True
                output_str=output_str+"\n</paragraph>\n"
                #replace multiple spaces with single space in output_str
                output_str=re.sub(' +', ' ', output_str) #useful, but we loose the ability to detect parsing errors
                #TODO decide whether to remove double spaces or not is a good idea
                
        output_str=output_str+"\n</page>\n"
    
    output_str=output_str+r"</document>"
    return output_str




def get_paragraphs(text, paragraph_start_tag="<paragraph>", paragraph_end_tag="</paragraph>"):
    """
    Extracts paragraphs from the input text that are enclosed by the specified start and end tags.

    Parameters
    ----------
    text :String
        The text containing the paragraphs to be processed.
    paragraph_start_tag :String, optional
        Indicates the start tag that encloses the paragraphs in the input text.
        The default is "<paragraph>".
    paragraph_end_tag :String, optional
        Indicates the end tag that encloses the paragraphs in the input text.
        The default is "</paragraph>".

    Returns
    -------
    paragraphs : list of strings
        A list of paragraphs in which paragraphs don't include initial or closing tags, only format tags.

    """
    ini_tag_offset=len(paragraph_start_tag)
    end_tag_offset=len(paragraph_end_tag)
    
    paragraphs=[]
    
    start_par_pos=text.find(paragraph_start_tag)
    
    while(start_par_pos!=-1):
        end_par_pos=text.find(paragraph_end_tag)
        #new paragraph
        p=text[start_par_pos+ini_tag_offset:end_par_pos-1]
        #print(p) #for testing purposes
        paragraphs.append(p)
        #update text by removing processed paragraph
        text=text[end_par_pos+end_tag_offset:]
        
        start_par_pos=text.find(paragraph_start_tag)
        
    return paragraphs

def get_first_opening_tag(text):
    '''
    Finds the first opening tag in the input text.

    Parameters
    ----------
    text :String
        Text to search for an opening tag.

    Returns
    -------
    String
        The first opening tag found processed without angle brackets, or if not tag is found or if the input text is empty.

    '''
    if len(text)>0:
        matches=re.findall(r"<[\w]+>", text)
        if len(matches)>0:
            return matches[0].replace("<","").replace(">","")
        else:
            print("No tag found")
            print(text)
            return "d'oh"
    return  "d'oh"

def get_first_opening_tag2(text):
    '''
    Finds the first opening tag in the input text.

    Parameters
    ----------
    text :String
        Text to search for an opening tag.

    Returns
    -------
    String
        The first opening tag found processed without angle brackets, or if not tag is found or if the input text is empty.

    '''

    #los for first "<" not followed by "/"
    #p = re.compile("<[^/][^>]*>")

    if len(text)>0:
        #matches=re.findall(r"<[\w]+>", text)
        matches=re.findall("<[^/][^>]*>", text)
        if len(matches)>0:
            return matches[0].replace("<","").replace(">","")
        else:
            print("No tag found")
            print(text)
            return "d'oh"
    return  "d'oh"
    
def get_last_closing_tag(text):
    '''
    Finds the last closing tag in the input text.

    Parameters
    ----------
    text : String
        Text to search for a closing tag..

    Returns
    -------
    String
        The last closing tag found and processed without angle brackets and forward slash, or if no tag is found

    '''
    matches=re.findall(r"</[\w]+>", text)
    if len(matches)>0:
        return matches[-1].replace("</","").replace(">","")
    else:
        return "d'oh"
    
def get_tags(text):
    """
    Returns two lists of the XML tags in the document (with and without delimiters)

    Parameters
    ----------
    text : String
        The text to extract tags from.

    Returns
    -------
    tags : list of strings
        Raw tags including the angle brackets: <tagname>.
    processed_tags : list of strings
        Tag names only, no delimiters.

    """

    # initial_tags=re.findall(r"<[^(/><.)]+>", text)
    # end_tags=re.findall(r"</[^(><.)]+>", text)
    initial_tags=re.findall(r"<[^( /><.)]+>", text) #inital tags cannot have '>', '/', ' ', or spaces inside
    end_tags=re.findall(r"</[^( ><.)]+>", text) #end tags cannot have '>', '/', ' ', or spaces inside
    
    tags=[]
    processed_tags=[]
    for t in initial_tags:
        tags.append(t)
        processed_tags.append(t[1:-1])
    for t in end_tags:
        tags.append(t)
        processed_tags.append(t[2:-1])

    tags=list(set(tags))
    processed_tags=list(set(processed_tags))
    
    return tags, processed_tags
        
def get_most_common_font_size(tags):
    '''
    Processes the input tags to extract font sizes among the input tags.

    Parameters
    ----------
    tags : List of strings
        Contains a list of tags to extract font sizes from.

    Returns
    -------
    most_common :Float
        Provides the most common font size among the input tags.

    '''
    sizes=[]
    for t in tags:
        f_size=get_trailing_number(t)
        if f_size!=None:
            sizes.append(f_size)
    most_common=max(statistics.multimode(sizes))
    return most_common
    
def get_paragraph_font_size(paragraph):
    '''
    Processes and finds the most common font size in the input paragraph.
    
    Parameters
    ----------
    paragraph : String
        The paragraph to extract the most common font size from.

    Returns
    -------
    s : Float
        Returns the most common fon size in the input paragraph.

    '''
    #remove opening and closing tags
    sanitized_txt=remove_tags(paragraph, ["paragraph"])
    #get remaining tags
    _,tags=get_tags(sanitized_txt)
    s=get_most_common_font_size(tags)
    return s
    
def get_text_between_tags(text, tag):
    '''
    finds the text enclosed by the specified tag in the input text.

    Parameters
    ----------
    text :String
        The text to search for the specified tag.
    tag :String
        Th tag enclosing the text to be extracted.

    Returns
    -------
    res :String
        The enclosed text by the specified tag in the input.
    '''
    
    #get initial tag position
    p_ini = re.compile("<"+tag+">")
    m = p_ini.search(text)
    ini_pos=m.start()+len("<"+tag+">")

    #get end tag position
    p_end = re.compile("</"+tag+">")
    m = p_end.search(text)
    end_pos=m.start()

    #get text in between tags
    res=text[ini_pos:end_pos]
    return res


def get_text_between_tags2(text, tag):
    '''
    Returns the text without tags, the text with tags included, and the initial position in the text for both cases

    Parameters
    ----------
    text : String
        The input text to search for the tag.
    tag : String
        The tag to search for in the text.

    Returns
    -------
    without_tags : String
        Text between the opening and closing tags.
    with_tags : String
        Text including the opening and closing tags.
    ini_tagless : Integer
        The starting position of the text between the tags.
    ini_tagged : Integer
        The starting position of the opening tag.

    '''

    #get initial tag position
    p_ini = re.compile(re.escape("<"+tag+">"))
    m_ini = p_ini.search(text)

    #get end tag position
    p_end = re.compile(re.escape("</"+tag+">"))
    m_end = p_end.search(text)
    

    #get text in between tags
    ini_pos=m_ini.start()+len("<"+tag+">")
    ini_tagless=ini_pos
    end_pos=m_end.start()
    without_tags=text[ini_pos:end_pos]

    #get text in between tags, tags includes    
    ini_pos=m_ini.start()
    ini_tagged=ini_pos
    end_pos=m_end.start()+len("</"+tag+">")
    with_tags=text[ini_pos:end_pos]
    
    return without_tags, with_tags, ini_tagless, ini_tagged 

def get_text_between_tags3(text, tag, verbose=False):
    '''
    Returns the text without tags, the text with tags included, and the initial position in the text for both cases

    Parameters
    ----------
    text : String
        The input text to search for the tag.
    tag : String
        The tag to search for in the text.

    Returns
    -------
    without_tags : String
        Text between the opening and closing tags.
    with_tags : String
        Text including the opening and closing tags.
    ini_tagless : Integer
        The starting position of the text between the tags.
    ini_tagged : Integer
        The starting position of the opening tag.

    '''

    print("text entering the function\n", text) if verbose==True else ""

    full_tag=tag

    #if tag starts with font
    if tag.startswith("font"):    
        tag=tag[:4]


    print("full tag:", full_tag) if verbose==True else ""
    print("tag:", tag) if verbose==True else ""


    #get initial tag position
    p_ini = re.compile(re.escape("<"+full_tag+">"))
    m_ini = p_ini.search(text)

    print("m_ini", m_ini) if verbose==True else ""

    #get end tag position
    p_end = re.compile(re.escape("</"+tag+">"))
    #m_end = p_end.search(text[m_ini.end():])
    m_end = p_end.search(text[m_ini.start():])
    

    #get text in between tags

    print("tag length", len("<"+full_tag+">")) if verbose==True else ""

    ini_pos=m_ini.start()+len("<"+full_tag+">")
    ini_tagless=ini_pos
   # end_pos=m_end.start()
    #end_pos=m_end.start()+len(text[m_ini.end():])
    end_pos=m_end.start()+m_ini.start()

    without_tags=text[ini_pos:end_pos]
 
    print("ini_pos", ini_pos) if verbose==True else ""
    print("end_pos", end_pos) if verbose==True else ""

    #get text in between tags, tags included    
    ini_pos=m_ini.start()
    ini_tagged=ini_pos
    #end_pos=m_end.start()+len("</"+tag+">")+len(text[m_ini.end():])
    end_pos=m_end.start()+m_ini.start()+len("</"+tag+">")

    with_tags=text[ini_pos:end_pos]
    
    return without_tags, with_tags, ini_tagless, ini_tagged 



# Ref: https://stackoverflow.com/questions/7085512/check-what-number-a-string-ends-with-in-python

def get_trailing_number(s):
    '''
    Finds the integer value of any digits that appear at the end of a string.

    Parameters
    ----------
    s : String
        The input string to search for trailing numbers/digits.

    Returns
    -------
    Integer or None
        Integer value of any digits that appear at the end of the input string.
        If there are not digits, it returns none.

    '''
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def get_images_from_pdfs(input_folder, output_folder):
    '''
    Converts each page of PDFs into images and saves them as JPEG files.

    Parameters
    ----------
    input_folder : String
        The folder contains the PDF files to be processed.
    output_folder : String
        The folder where the image files will be stored.

    Returns
    -------
    None.

    '''
    
    files=os.listdir(input_folder)
    for f in files:
        name_base=f[:-4]
        pdf_path=join(input_folder,f)
        pdf_path_aux=pdf_path.lower()
        print("File to be processed: "+pdf_path)
        if pdf_path_aux.endswith(".pdf"):
            images=convert_from_path(pdf_path)
            #get images
            max_page=len(images)
            #max_page=10
            for i in range(max_page):
                image_filename=join(output_folder,name_base+"_"+str(i)+".jpg")
                images[i].save(image_filename, "JPEG")
                
def remove_tags(text, tags_list):
    '''
    Removes specified tags from the input text.

    Parameters
    ----------
    text : String
        The input text tags from the input text.
    tags_list : list of strings
        A list of tags to be processed from the input text.

    Returns
    -------
    text : String
        The text with the specified tags removed.

    '''
    for tag in tags_list:
        text=text.replace("<"+tag+">", "").replace("</"+tag+">", "")
    return text
                
def sanitized_xml_from_pdf(file_name, font_db_conn):
    '''
    Retrieves and cleans up the text from a PDF file and returns it in an XML-format.

    Parameters
    ----------
    file_name : String
        PDF file's name to be processed.
    font_db_conn :Object
        A connection to a font database to find the label for each font type and size.

    Returns
    -------
    sanitized_xml :string
        A string containing the sanitized text content of the PDF file in an XML-format.

    '''
    
    
    xml=pdf_to_labeled_text(file_name, font_db_conn)
    #clean up xml
    xml=clean_up_xml(xml)
    
    paragraphs=get_paragraphs(xml)
    new_paragraphs=detect_paragraph_anomalies(paragraphs, xml)
    #new_paragraphs=paragraphs
    sanitized_xml=paragraphs_to_text(new_paragraphs)
    
    # tags=["page"]
    # xml=remove_tags(xml, tags)
    sanitized_xml="<document>\n"+sanitized_xml+"\n</document>"
    return sanitized_xml
                
def xml_to_raw_text_file(xml_text, output_file_name, verbose=False):
    '''
    Extracts the tags from XML-format, removes them, and saves the results to a file

    Parameters
    ----------
    xml_text : String
        DESCRIPTION.
    output_file_name : String
        DESCRIPTION.
    verbose : Bool, optional
        If set to True, prints the information about the tags in the text and indicates the file is saved successfully.
        The default is False.

    Returns
    -------
    None.

    '''
    #get tags
    _, tags = get_tags(xml_text)
    print("Tags in document: "+str(tags)) if verbose==True else ""
    #remove tags
    text=remove_tags(xml_text, tags)
    #save new file
    f = open(output_file_name, "w")
    f.write(text)
    f.close()
    print("File saved successfully")
    
def xml_to_raw_text(xml_text):
    '''
    Extracts the tags from an XML file and removes them.

    Parameters
    ----------
    xml_text : String
        The input XML text to be processed.

    Returns
    -------
    text : String
        The input XML text without tags.

    '''
    #get tags
    _, tags = get_tags(xml_text)
    #remove tags
    text=remove_tags(xml_text, tags)
    return text
    
#TODO see where xml_to_raw_text is being used to change it to xml_to_text_file


def paragraphs_to_text(paragraphs):
    '''
    Concatenates the paragraphs and encloses each string withing paragraph tags.

    Parameters
    ----------
    paragraphs : List of strings
        List of string to join.

    Returns
    -------
    text : String
        The joined text with paragraph tags added around each string.

    '''
    text=""
    for p in paragraphs:
       text=text+"<paragraph>"+p+"\n</paragraph>\n"
    return text


def pdf_file_to_sau(file_name, target_folder, font_db_conn, paragraphs_df=None, styles_df=None, documents_df=None):
    aux_ind=file_name.rfind("/")
    raw_name=file_name[aux_ind+1:]
    extensionless_name=raw_name[:raw_name.rfind(".")]
    doc_name=extensionless_name.upper()
    sanitized_file=target_folder+"/sanitized_"+extensionless_name+".xml"
    doc_info=doc_name.split("-")

    if doc_name.startswith("T") or doc_name.startswith("C"):
        doc_type=doc_info[0]
        doc_year=doc_info[2]
        
    elif doc_name.startswith("SU") or doc_name.startswith("A"):
        doc_year=doc_info[1]
        if doc_name.startswith("SU"):
            doc_type="SU"
        else:
            doc_type="A"
    else:
        print("Document type not recognized")

    #convert year to 4 digits
    if len(doc_year)==2:
        if int(doc_year)>90:
            doc_year="19"+doc_year
        else:
            doc_year="20"+doc_year
    #convert year to integer
    doc_year=int(doc_year)
        
    xml_text=kpdf.sanitized_xml_from_pdf(file_name, font_db_conn)
    output_f = open(sanitized_file, "w")
    output_f.write(r""+xml_text)
    output_f.close()

    #read xml file
    with open(sanitized_file, 'r') as file:
        data = file.read()

    #check if documents_df exists
    if documents_df is not None:
        #check if there exist a document with the same name in the dataframe
        if len(documents_df.loc[documents_df["name"]==extensionless_name])==0: #if there isn't any
            #add new entry to documents_df dataframe
            documents_df.loc[len(documents_df)] = [extensionless_name, doc_type, doc_year]
            doc_id=len(documents_df)-1
        else:
            #get document id
            doc_id=documents_df.loc[documents_df["name"]==extensionless_name]["id"].values[0]
            
    else:
        #create new dataframe with columns id, name, sentence_id is the index
        documents_df=pd.DataFrame(columns=["id", "name", "type", "year"])
        documents_df.set_index('id', inplace=True)
        #add new entry to documents_df dataframe
        documents_df.loc[len(documents_df)] = [extensionless_name, doc_type, doc_year]
        doc_id=len(documents_df)-1

    styles_df, paragraphs_df, sections, xml=sau.get_semantic_blocks_from_XML(data,doc_id, paragraphs_df, styles_df)

    semantic_f = open(target_folder+"/semantic_"+extensionless_name+".xml", "w")
    semantic_f.write(xml)
    semantic_f.close()

    return paragraphs_df, styles_df, documents_df

def pdf_files_to_sau(input_folder, output_folder, font_db, paragraphs_df=None, styles_df=None, documents_df=None):

    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")    
    files=os.listdir(py_input_folder)

    for f in files:
        file_path=join(input_folder,f)
        print("File to be processed: "+file_path)
        if file_path.endswith(".pdf"):
            paragraphs_df, styles_df, documents_df=pdf_file_to_sau(file_path, output_folder, font_db, paragraphs_df, styles_df, documents_df)
    
    return paragraphs_df, styles_df, documents_df


def get_tag_instances(tag, text, num_instances):

    if tag.startswith("font"):
        #full_tag=tag
        tag=tag[:4]

    #instances=[]

    res=""
    p = re.compile("<"+tag)
    m = p.search(text)
    num_instances_found=0
    while m!=None and num_instances_found<num_instances:
        num_instances_found=num_instances_found+1
        #instances.append(m.group())
        #instances=instances+m.group()
        m = p.search(text, m.end())
    
    
    p1 = re.compile("</"+tag+">")
    m1 = p1.search(text, m.end())

    res=text[m.start():m1.end()]
    return res