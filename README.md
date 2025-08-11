**📌 Retrieval-Based Q&A Assignment (LangChain)**
**📄 Overview**
This repository contains solutions for 4 tasks involving LangChain, document loading, text splitting, and RetrievalQA.
The goal is to demonstrate building basic RAG (Retrieval-Augmented Generation) pipelines in Python.

**📝 Assignment Tasks**

1.Simple RAG Pipeline

Build a basic Retrieval-Augmented Generation pipeline that performs Q&A over a document.

2.Text Generation

Generate short, relevant answers to given prompts without complex sampling parameters.

3.RetrievalQA with Custom Policy Document

Load a sample company policy document and answer the question:
"What is the refund policy?" using RetrievalQA.

4.Document Loading & Splitting

Load a .txt or .pdf file using LangChain's TextLoader or PyPDFLoader.

Split it using RecursiveCharacterTextSplitter and print the total number of chunks created.

**📂 Repository Structure**

.
├── Ass_51.py             # Task 1
├── Ass_52_OceanPoem.py   # Task 2
├── Ass_53_QA.py          # Task 3
├── Ass_54_SC.py          # Task 4
└── README.md

**⚙️ Requirements**
Install dependencies:
pip install langchain langchain-community langchain-text-splitters pypdf

If you are using HuggingFace models:
pip install transformers


**📌 Notes**
For .pdf files, use PyPDFLoader.

For .txt files, use TextLoader with encoding="utf-8".

Ensure you have the document files in the same directory as the script or update the file path.

