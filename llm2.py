from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import giskard

from giskard.llm.client.bedrock import ClaudeBedrockClient

import os

bedrock_runtime = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])

# Wrap the Beddock client with giskard Bedrock client and embedding
claude_client = ClaudeBedrockClient(bedrock_runtime, model="anthropic.claude-3-sonnet-20240229-v1:0")

giskard.llm.set_default_client(claude_client)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
documents = loader.load_and_split(text_splitter)

# Use HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, hf_embeddings)

# Define the prompt template
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant.
Your task is to answer questions based on the IPCC Climate Change Report (2023).
Be concise and informative.

Context:
{context}

Question:
{question}

Your answer:
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Initialize ChatBedrock with the correct model ID
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0},
    client=boto3.client("bedrock-runtime", region_name="ap-south-1")
)

# Create the RetrievalQA chain
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Test the QA chain with a sample question
question = "What are the impacts of climate change on agriculture?"
response = climate_qa_chain.run(question)
print(response)