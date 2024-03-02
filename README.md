# Automated Annotation of Court Decisions
## Colombian Constitutional Court

This system processes Colombian Constitutional Court's decisions and classifies the text by section and type of text (regular, quote, emphasis, footnote, etc.)

Using some of the functions developed for this endeavor, the user can do the following:

1) Convert RTF files to PDF format.
2) Convert a PDF file into an XML with tags denoting pages, paragraphs, and text blocks (with format information). The system has advanced capabilities to fix some errors related to parser interpretation of blank spaces
3) Create a database of all font types (font name + size) used in a collection of documents.
4) Analyze a collection of PDF judicial decisions following the conventions used by the Colombian Constitutional Court and split them in paragraphs and text blocks, assigning to each a section
5) Ask chatGPT or Llama-based systems to classify both paragraphs and text blocks assigning sections and intended purpose.

How to run

To run the full system, it is necessary to accomplish three tasks:
1) Convert the PDF files to Pandas dataframes
   To process an entire folder, run the function 
3) Do the text analysis and assign sections
   T
5) Compare the system annotations to LLMs' by querying chatGPT and Llama



Files in the repository:
kutils.py: Various functions to process text from
