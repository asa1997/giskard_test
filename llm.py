from langchain import Bedrock, FAISS, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Prepare the vector store (FAISS) with the IPCC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
documents = loader.load_and_split(text_splitter)

# Step 2: Use Hugging Face embeddings instead of OpenAIEmbeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, hf_embeddings)

# Step 3: Prepare the QA chain with Claude via Bedrock
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant.
Your task is to answer questions on climate change using the provided context from the IPCC report.
Please provide clear and concise answers based on the context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

# Step 4: Initialize Claude using Amazon Bedrock
llm = Bedrock(model="claude-2", model_kwargs={"temperature": 0})
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Example usage
question = "What are the impacts of climate change on agriculture?"
result = climate_qa_chain.run(question)
print(result)
