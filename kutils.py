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
    t_name : str
        The table's name to be created.
    conn : Object
        A SQLite database connection.
    fields : str
        List of field names for the table.
    field_types : str
        A list of field types for the table. Its length should be the same as 'fields'.

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
    """
    Connects to a SQLite database, checks for a specified table's existence, and creates it with given fields if it doesn't exist.

    Parameters
    ----------
    db_name : str
        Database file name.
    t_name : str
        Target table name.
    fields : list of str
        Field names for table creation. The default is [].
    field_types : list of str
        Data types for each field. The default is [].

    Returns
    -------
    conn: sqlite3.Connection
       Connection object to the database.
    """
    
    conn=sl.connect(db_name)
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

def connect_to_font_db(fonts_db="", fonts_table="", font_fields=[], font_field_types=[]):
    """
    Connects to a SQLite font database and creates a table with default or specified fields. 

    Parameters
    ----------
    fonts_db : str
        Font database filename.
    fonts_table : str
        Font table name.
    font_fields : list of str
        Field names for the font table:["name", "size", "label"]
    font_field_types : list of str
        Data types for the fields:["TEXT", "REAL", "TEXT"].

    Returns
    -------
    font_db_conn : sqlite3.Connection
        Connection to the font database.
    """
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

def add_db_entry(conn, t_name, fields_to_fill, field_values):
    """
    Inserts a new entry into a specified table within a SQLite database using provided field names and values.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.
    t_name : str
        Name of the table to insert the entry into.
    fields_to_fill : list of str
        List of field names for the entry.
    field_values : list
        Corresponding values for the fields.

    """
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
    
    
    c.execute(query, tuple(field_values) )
    conn.commit()


def connect_to_font_db(fonts_db="", fonts_table="", font_fields=[], font_field_types=[]):
    """
    Returns a connection to a font database. It creates the database if it can't find it.

    Parameters
    ----------
    fonts_db : TYPE, optional
        DESCRIPTION. The default is "".
    fonts_table : TYPE, optional
        DESCRIPTION. The default is "".
    font_fields : TYPE, optional
        DESCRIPTION. The default is [].
    font_field_types : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    font_db_conn : TYPE
        DESCRIPTION.

    """
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
    main_folder : str
        The location of the zip files
    output_directory : String
        The location of the unzip files

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

    """
    folders=[x[0] for x in os.walk(input_folder)]
    for folder in folders:
        files = os.listdir(folder)
        for f in files:
            file_path=join(folder,f)
            if isfile(file_path) and file_path.endswith(".rtf"):
                shutil.move(file_path, target_folder+"/"+f)
                

def convert_rtf_files(libreoffice_path, input_folder, output_folder):
    """
    Converts RTF files in input_folder into PDF files and saves them 
    to output folder. If a file with the same name already exists in the
    output folder, it will be omitted.

    It requires Libreoffice to be installed to work.

    Parameters
    ----------
    libreoffice_path : str
       LibreOffice's path 
    input_folder : str
        Location of the RTF files to convert
    output_folder : str
        Output folder of the PDF files

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
    text : str
        The input text to be analyzed.

    Returns
    -------
    uppercase_prop : Float
        The proportion of characters in the text that are uppercase letters.
    up_count : int
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
    text : str
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


def detect_roman_numerals_by_format(text):
    '''
    Detects Roman numerals in three formats: pure, number followed by a dot, 
    and number followed by a hyphen.

    Parameters
    ----------
    text : str
        The input text as a string.

    Returns
    -------
    pure_roman_numerals : str
        List of detected pure Roman numerals.
    dot_roman_numerals : str
        List of detected dot-formatted Roman numerals.
    hyphen_roman_numerals : str
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
    text : str
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
    Finds Arabic numerals following two patterns: 1) number + hyphen or 2) number + dot
    
    Parameters
    ----------
    text : str
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


def detect_alphabetical_numbering_by_format(text):
    '''
    Finds Arabic numerals following two patterns: 1) number + space (optional) + hyphen 
    or 2) number +space(optional) + dot.
    
    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    -------
    hyphen_matches : List
        A list of substrings from the input text that match the hyphen pattern.
    dot_matches : List
        A list of substrings from the input text that match the dot pattern.

    '''

    hyphen_pattern=r"^[A-Z][ ]*[-][ ]*"
    hyphen_matches=re.findall(hyphen_pattern, text)

    dot_pattern=r"^[A-Z][ ]*[.][ ]*"
    dot_matches=re.findall(dot_pattern, text)

    return hyphen_matches, dot_matches

def detect_numerals_by_format(text):
    '''
    
    Given a string, it looks for numbers that follow the pattern roman/arabic number + dash/dot.
    It returns the name of the pattern found and the first number found within the string.


    Parameters
    ----------
    text : str
        The input text to be analyzed.

    Returns
    -------
    num_format : str
       A string indicating the format of the numerals present in the text.
    number : str
       The first number found within the text that follows the format detected.

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
    Adjusts spacing around periods and hyphens in a given string to ensure consistent formatting.
    
    Parameters
    ----------
    text : str
        The input text to be formatted.

    Returns
    -------
    text :str
        The formatted text with corrected spacing.

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
    text_block : str
        Test to be analyzed.
    words : List
        List of categories.
    associated_terms : List
        List of lists with the associated terms for each word.

    Returns
    -------
    res : str
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


def roman_to_int(s):
    """
    Converts a Roman numeral to its integer equivalent.

    Parameters
    ----------
    num : str
        Roman numeral string. 

    Returns
    -------
    roman_num : int.
       Integer value of the Roman numeral; returns -2 if the input contains invalid Roman numeral characters.
    """

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
    """
    Converts an integer to its Roman numeral representation.
    
    Parameters
    ----------
    s : int
        The integer to convert.

    Returns
    -------
    str
        The Roman numeral representation of the given integer.
    """
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
    """
    Given a number (Roman or Arabic) it returns the number +1 in regular (arabic) representation.

    Parameters
    ----------
    number : str
        The number as a string.

    Returns
    -------
    int
        The incremented number if valid; -2 if invalid.
    """
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
    else: 
        return -2



def get_word_cloud_from_df(df, column_name, stopwords=[], max_words=100, max_font_size=50, background_color="black", width=800, height=400):
    """
    Creates a word cloud from a DataFrame column, allowing customization of stopwords, word count, font size, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column_name : str
        Column for word cloud data.
    stopwords : list, optional
        Words to exclude. The default is [].
    max_words : int, optional
        Maximum word count. The default is 100.
    max_font_size : int, optional
        The default is 50.
    background_color : str, optional
        The default is "black".
    width : int, optional
        The default is 800.
    height : int, optional
        The default is 400.

    """

    word_dict = df.to_dict()[column_name]
    word_freq_dict = dict(zip(df['word'], df['frequency']))
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

    wordcloud = WordCloud(stopwords=stopwords, max_words=max_words, max_font_size=max_font_size, background_color=background_color, width=width, height=height).generate_from_
    plt.imshow(wordcloud, interpolation='bilinear')