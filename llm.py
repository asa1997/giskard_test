from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import json
import giskard
import pandas as pd
# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
db = FAISS.from_documents(loader.load_and_split(text_splitter))

# Prepare QA chain
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

bedrock = boto3.client(
  service_name='bedrock-runtime',
  region_name='ap-south-1'
)

modelId = 'mistral.mistral-7b-instruct-v0:2'
body = json.dumps({
    "temperature": 0.5,
})

llm = bedrock.invoke_model(
    modelId=modelId,
    body=body
)
# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Test that everything works
climate_qa_chain.invoke({"query": "<s>[INST] Is sea level rise avoidable? When will it stop?[INST]"})




def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [climate_qa_chain.invoke({"query": question}) for question in df["question"]]


# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=["question"],
)

# Optional: let’s test that the wrapped model works
examples = [
    "According to the IPCC report, what are key risks in the Europe?",
    "Is sea level rise avoidable? When will it stop?",
]
giskard_dataset = giskard.Dataset(pd.DataFrame({"question": examples}), target=None)

print(giskard_model.predict(giskard_dataset).prediction)