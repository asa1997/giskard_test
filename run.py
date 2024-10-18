from langchain import OpenAI, FAISS, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import giskard
import pandas as pd

# Prepare vector store (FAISS) with IPPC report
from transformers import pipeline
from langchain import FAISS, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  # Assuming this exists or you create a similar interface
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming modifications for HuggingFaceEmbeddings and other necessary adjustments

# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")

# Assuming you have a way to convert documents to embeddings using HuggingFaceEmbeddings
db = FAISS.from_documents(loader.load_and_split(text_splitter), HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

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

# Use Hugging Face's pipeline for question answering
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(context, question):
    return qa_pipeline(question=question, context=context)["answer"]

# Assuming RetrievalQA can be adapted to use a custom function for answering
climate_qa_chain = RetrievalQA.from_custom_function(custom_function=answer_question, retriever=db.as_retriever(), prompt=PROMPT_TEMPLATE)


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

scan_results = giskard.scan(giskard_model)

# display(scan_results)
print("======Scan results=======,")