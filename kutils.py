#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 4 10:56:46 2023
Updated on Thu Oct 19 2023

@author: KarolinaNaranjo
"""

import os
from os.path import isfile, join
import subprocess
import zipfile
import shutil
import sqlite3 as sl

import re

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def create_db_table(t_name, conn, fields, field_types):
    '''
    Creates a new table in a SQLite database.

    Parameters
    ----------
    t_name :String
        The table's name to be created.
    conn : Object
        A SQLite database connection.
    fields : String
        List of field names for the table.
    field_types : String
        A list of field types for the table. Their length should be the same as 'fields'.

    Returns
    -------
    bool
        True if the table is successfully created.

    '''
    
    with conn:
        
        query="CREATE TABLE "+t_name+"""  (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT"""
        for i, field in enumerate(fields):
            query=query+",\n" +field+" " + field_types[i].upper()
            
        query=query+");"
        
        print(query)
        
        conn.execute(query)
        conn.commit()
    return True

def connect_to_db(db_name, t_name, fields=[], field_types=[]):
    
    conn=sl.connect(db_name) #connect to database
    c=conn.cursor()
    
    #conn.set_trace_callback(print) #uncomment to see database queries - DO NOT ERASE

    c.execute( "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='"+t_name+"' ")

    
    if c.fetchone()[0]==1:
        print('Table already exists.')
    else:
        print("Table doesn't exist. It will be created now.")
        create_db_table(t_name, conn, fields, field_types)

    conn.commit()
    return conn

def add_db_entry(conn, t_name, fields_to_fill, field_values):
    c=conn.cursor()
    
    query1="INSERT INTO "+t_name+"("
    query2="VALUES ("
    for i, field in enumerate(fields_to_fill):
        if i!=0:
            query1=query1+",\n"
            query2=query2+","
        query1=query1+field
        query2=query2+"?"
        
    query=query1+") " + query2 + ")"
    
    #print(query)
    
    c.execute(query, tuple(field_values) )
    conn.commit()


def connect_to_font_db(fonts_db="", fonts_table="", font_fields=[], font_field_types=[]):
    if fonts_db=="":
        fonts_db="fonts_db.db"

    if fonts_table=="":
        fonts_table="fonts"
    if font_fields==[]:
        font_fields=["name", "size", "label"]
    if font_field_types==[]:
        font_field_types=["TEXT", "REAL", "TEXT"]

    font_db_conn=connect_to_db(fonts_db, fonts_table, font_fields, font_field_types)
    return font_db_conn


def unzip_folders(main_folder, output_directory):
    """
    Unzips all zip files in main_folder. It extracts them to output_directory

    Parameters
    ----------
    main_folder : String
        The location of the zip files
    output_directory : String
        The location of the unzip files

    Returns
    -------
    None.

    """
    for filename in os.listdir(main_folder):
        if filename.endswith(".zip"):
            print(filename)
            with zipfile.ZipFile(main_folder+"/"+filename, 'r') as zip_ref:
                zip_ref.extractall(output_directory)
                

def subfolder_contents_to_folder(input_folder, target_folder):
    """
    Move all subfolders' files to a specific folder (collapse tree structure)

    Parameters
    ----------
    input_folder :String
        Folder tree to be collapsed
    target_folder :String
        Output folder

    Returns
    -------
    None.

    """
    folders=[x[0] for x in os.walk(input_folder)]
    for folder in folders:
        files = os.listdir(folder)
        for f in files:
            #print(join(folder,f))
            file_path=join(folder,f)
            if isfile(file_path) and file_path.endswith(".rtf"):
            #     print(f)
            #     file_name = os.path.join(source, file)
                shutil.move(file_path, target_folder+"/"+f)
                

def convert_rtf_files(libreoffice_path, input_folder, output_folder):
    """
    Converts RTF files in input_folder into a PDF files and saves them 
    to output folder. If a file with the same name already exists in the
    output folder, it will be omitted.

    Parameters
    ----------
    libreoffice_path : String
       LibreOffice's path 
    input_folder :String
        Location of the RTF files to convert
    output_folder :String
        Output folder of the PDF files

    Returns
    -------
    None.
    """
    print("shell path: "+ input_folder)
    #format path for python use
    py_input_folder=input_folder.replace(r"\~", "~").replace(r"\ ", " ")
    py_output_folder=output_folder.replace(r"\~", "~").replace(r"\ ", " ")
    print("python path: "+py_input_folder)
    files=os.listdir(py_input_folder)
    for f in files:
        file_path=join(input_folder,f)
        print(file_path)
        if file_path.endswith(".rtf"):
            output_file_path=join(py_output_folder, f.replace(".rtf", ".pdf"))
            if isfile(output_file_path)==False:
                cmd='./soffice --headless --convert-to pdf '+file_path+' --outdir '+output_folder
                #print(cmd)
                subprocess.run(cmd, shell=True, cwd=libreoffice_path)
            else: 
                print("There is already a file with the same name in the destination folder.")
    return ""

def count_uppercase(text):
    '''
    Counts the number of uppercase letters in a given text 
    and calculates the proportion of characters that are uppercase.

    Parameters
    ----------
    text : Str
        The input text to be analyzed.

    Returns
    -------
    uppercase_prop : Float
        The proportion of characters in the text that are uppercase letters.
    up_count : Integer
        The total number of uppercase letters in the text.

    '''
    up_count=0
    for c in text:
        if c.isupper():
            up_count+=1
    
    uppercase_prop=up_count/len(text)
    return uppercase_prop, up_count

def detect_roman_numerals(text):
    '''
    Takes a string and outputs a list of words composed solely of 
    characters that represent Roman numerals.
    
    Parameters
    ----------
    text : String
        The input text to be analyzed.

    Returns
    -------
    roman_numerals : List
        A list of words from the input text that contain only characters representing Roman numerals.
    '''
    numerals = ['M', 'D', 'C', 'L', 'X', 'V', 'I']
    roman_numerals = []
    words = text.split()
    for word in words:
        if all(char in numerals for char in word):
            roman_numerals.append(word)
    return roman_numerals

#TODO Maybe we can write a smarter function for detecting roman numerals, one in which we actually check that the number makes sense, not just that it has the right characters.


def detect_roman_numerals_by_format(text):
    '''
    Detects Roman numerals in three formats: pure, number followed by a dot, 
    and number followed by a hyphen.

    Parameters
    ----------
    text : String
        The input text as a string.

    Returns
    -------
    pure_roman_numerals : String
        List of detected pure Roman numerals.
    dot_roman_numerals : String
        List of detected dot-formatted Roman numerals.
    hyphen_roman_numerals : String
        List of detected hyphen-formatted Roman numberals.

    '''
    numerals = ['M', 'D', 'C', 'L', 'X', 'V', 'I']
    text=text.replace(".", ". ")
    text=text.replace(" .", ".")
    text=text.replace(" -", "-")
    text=text.replace("-", "- ")
    words = text.split()

    pure_roman_numerals = []
    dot_roman_numerals=[]
    hyphen_roman_numerals=[]
    for word in words:
        if all(char in numerals for char in word):
            pure_roman_numerals.append(word)
        elif (all(char in numerals for char in word[:-1])and word[-1]==".") or (all(char in numerals for char in word[:-2])and word[-2:]==".-") :
            dot_roman_numerals.append(word)
        elif all(char in numerals for char in word[:-1])and word[-1]=="-":
            hyphen_roman_numerals.append(word)
        
    return pure_roman_numerals, dot_roman_numerals, hyphen_roman_numerals


def detect_numerals(text):
    '''
    Finds numerals in the text as list of substrings.
    
    Parameters
    ----------
    text :String
        The input text to be analyzed.

    Returns
    -------
    numerals :List
        A list of substrings from the input text that match numeral's pattern.

    '''
    pattern = r"\b\d+\b[\.-]"  # Matches one or more digits
    numerals = re.findall(pattern, text)
    return numerals

def detect_arabic_numerals_by_format(text):
    '''
    Finds Arabic numerals by identitying patterns of hyphen and dots used in the text.
    
    Parameters
    ----------
    text : String
        The input text to be analyzed.

    Returns
    -------
    hyphen_numerals : List
        A list of substrings from the input text that match the hyphen pattern.
    dot_numerals : List
        A list of substrings from the input text that match the dot pattern.

    '''
    hyphen_pattern = r"\b\d+\b[-]"  # Matches one or more digits
    hyphen_numerals = re.findall(hyphen_pattern, text)
    
    dot_pattern = r"\b\d+\b[\.]"  # Matches one or more digits
    dot_numerals = re.findall(dot_pattern, text)
    
    return hyphen_numerals, dot_numerals

#TODO we should consider implementing a detection type for numerals of the form #.#.#.

def detect_alphabetical_numbering_by_format(text):
    hyphen_pattern=r"^[A-Z][ ]*[-][ ]*"
    hyphen_matches=re.findall(hyphen_pattern, text)

    dot_pattern=r"^[A-Z][ ]*[.][ ]*"
    dot_matches=re.findall(dot_pattern, text)

    return hyphen_matches, dot_matches

def detect_numerals_by_format(text):
    '''
    Receives a string and returns a string that characterizes the format of the numerals contained within the text.

    Parameters
    ----------
    text : String
        The input text to be analyzed.

    Returns
    -------
    num_format : String
       A string indicating the format of the numerals present in the text.

    '''
    format_found=False
    ara_hyp_num, ara_dot_num=detect_arabic_numerals_by_format(text)
    if len(ara_hyp_num)>0:
        num_format="ara-hyphen"
        format_found=True
        number=ara_hyp_num[0].replace("-", "")
    elif not format_found and len(ara_dot_num)>0:
        num_format="ara-dot"
        format_found=True
        number=ara_dot_num[0].replace(".", "")
    else:
        rom_num, rom_dot_num, rom_hyp_num=detect_roman_numerals_by_format(text)
        if len(rom_hyp_num)>0:
            num_format="rom-hyphen"
            format_found=True
            number=rom_hyp_num[0].replace("-", "")
        elif len(rom_dot_num)>0:
            num_format="rom-dot"
            format_found=True
            number=rom_dot_num[0].replace(".", "")
    if format_found == False:
        #check if the format is alphabetical
        alpha_hyp_num, alpha_dot_num=detect_alphabetical_numbering_by_format(text)
        if len(alpha_hyp_num)>0:
            num_format="alpha-hyphen"
            format_found=True
            number=alpha_hyp_num[0].replace("-", "")
        elif len(alpha_dot_num)>0:
            num_format="alpha-dot"
            format_found=True
            number=alpha_dot_num[0].replace(".", "")
        else:
            num_format="unknown"
            number="-1"
    
    return num_format, number


def most_common_item(lst):
    '''
    Takes a list as input and returns the most common element in the list.
    Parameters
    ----------
    lst : List
        The input list to be analyzed.

    Returns
    -------
    most_common_value :Any 
        The most common element in the input list.

    '''
    if len(lst)==0:
        print("The list is empty.")
        return None
    counter = Counter(lst)
    most_common_value = counter.most_common(1)[0][0]
    
    return most_common_value

def clean_up_spaces(text):
    '''
    Cleans up spacing around periods and hyphens.

    Parameters
    ----------
    text : String
        The input text to be cleaned up.

    Returns
    -------
    text : TYPE
        DESCRIPTION.

    '''
    text=text.replace(".", ". ")
    text=text.replace(" .", ".")
    text=text.replace(" -", "-")
    text=text.replace("-", "- ")
    text=re.sub(' +', ' ', text)

    return text

def get_mentions(text_block, words, associated_terms, case=False):
    """
    Given a search term (text_block), a list of possible categories it can belong to (words), 
    and a list of terms associated to each category (associated_terms), 
    the function finds the category that corresponds to the search term.

    Parameters
    ----------
    text_block : String
        Test to be analyzed.
    words : List
        List of categories.
    associated_terms : List
        List of lists with the associated terms for each word.

    Returns
    -------
    res : String
        Text's category.

    """

    found=False
    res="Pending"
    if case==False:
        text_block=text_block.lower()
    
    i=0
    while i<len(words) and not found:

        j=0
        while j<len(associated_terms[i]) and not found:
            d=associated_terms[i][j]
            aux=re.search(r""+d, text_block)
            if aux:
                res=words[i]
                found=True
            j+=1
        i+=1
    return res

# def remove_all_instances(lst, item):
#     res_set=set(lst).difference({item})
#     return list(res_set)

def roman_to_int(s):
    #TODO work on creating a version that actually enforces the rules of roman numerals

    #check if the string only has the characters I, V, X, L, C, D, M
    if not all(char in ['I', 'V', 'X', 'L', 'C', 'D', 'M'] for char in s):
        return -2

    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val

def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while  num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def next_number(number):
    #remove spaces, periods, and hyphens
    number=number.replace(" ", "")
    number=number.replace(".", "")
    number=number.replace("-", "")
    
    print("number: "+number + " type: "+str(type(number)))
    #if it's a roman numeral
    if number.isupper():
        #return int_to_roman(roman_to_int(number)+1)
        return roman_to_int(number)+1
    elif number.isdigit(): #check if it's an arabic numeral
        return int(number)+1
    else: #if it's not a number, return -2
        return -2



def get_word_cloud_from_df(df, column_name, stopwords=[], max_words=100, max_font_size=50, background_color="black", width=800, height=400):

    # Create a dictionary from 'verb' and 'frequency' columns
    word_freq_dict = dict(zip(df['verb'], df['frequency']))
    
    all_stopwords = set(STOPWORDS)
    if len(stopwords)>0:
        all_stopwords.update(stopwords)
    
    #generate word cloud
    wordcloud = WordCloud(stopwords=all_stopwords, max_words=max_words, max_font_size=max_font_size, background_color=background_color, width=width, height=height).generate_from_frequencies(word_freq_dict)
    #plot word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    #plt.show()