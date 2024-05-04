
import os 
from dotenv import load_dotenv

load_dotenv()#

os.environ["OPENAI_API_KEY"] =  str(os.getenv("OPENAI_API_KEY"))
from langchain.llms import OpenAI

import pypdf
from pypdf import PdfReader

# Example usage:
pdf_address = "C:\\Users\\ashvi\\Jupyter Notebooks\\Work Project\\chroma_pdf_dir\\7-11-17-334.pdf"
try:
  reader = PdfReader(pdf_address)
  print("PDF loaded successfully!")
except FileNotFoundError:
  print("Error: The specified PDF file was not found.")


pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

#print(pdf_texts[0])

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

#print(character_split_texts[10])
print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(token_split_texts[10])
print(f"\nTotal chunks: {len(token_split_texts)}")

import chromadb

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()
#print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("Food_Sci_Db", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

def retrieve_documents(query):
  results = chroma_collection.query(query_texts=[query], n_results=5)
  retrieved_documents = results['documents'][0]

#  for document in retrieved_documents:
#      print(document)
#      print('\n')

  return retrieved_documents

query_topic = str(input("Enter the relevant topic for your query"))
retrieve_documents(query_topic)

import openai
from langchain_community.llms import OpenAI
from openai import OpenAI

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()

def rag(query, retrieved_documents, model="gpt-3.5-turbo"):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert food research assistant. Your users are asking questions about information contained in research journals."
            "You will be shown the user's question, and the relevant information from the journal. Answer the user's question using only this information. Always output in JSON format."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

query = str(input("Enter your query"))
docs = retrieve_documents(query)
#print(docs)
output = rag(query=query, retrieved_documents=docs)

print(output)

print("SOURCES:")
print("--------------------------------------------")

for i, string_value in enumerate(docs, start=1):
    print(f"source {i} : {string_value} \n")
    print ("--------------------------------\n")

