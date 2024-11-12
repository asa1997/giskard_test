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

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [climate_qa_chain.run({"query": question}) for question in df["question"]]

# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=["question"],
)

print("###########Running Scan####################")
scan_results = giskard.scan(giskard_model)


display(scan_results)

# Or save it to a file
scan_results.to_json("scan_results_claude.json")