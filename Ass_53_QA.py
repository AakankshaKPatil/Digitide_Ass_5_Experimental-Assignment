# retrieval_qa_offline.py
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Create sample document
sample_policy = """Company Policy:
1. Refund Policy: Customers may request a full refund within 30 days of purchase
   if the product is returned in original condition with proof of purchase.
2. Exchange Policy: Exchanges are allowed within 45 days for items of equal value.
3. Warranty: All products come with a 1-year warranty against manufacturing defects.
"""
with open("company_policy.txt", "w") as f:
    f.write(sample_policy)

# 2. Load document
loader = TextLoader("company_policy.txt")
documents = loader.load()

# 3. Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Load local model into HuggingFacePipeline
model_name = "google/flan-t5-base"  # Small, good for QA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 5. Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 6. Ask question
query = "What is the refund policy?"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)
