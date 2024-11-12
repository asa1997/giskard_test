from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence
import os
import json
import giskard
from giskard.llm.client import set_default_client
from giskard.llm.config import LLMConfigurationError
from giskard.llm.errors import LLMImportError
from giskard.llm.client.base import LLMClient, ChatMessage
import pandas as pd

class BaseBedrockClient(LLMClient, ABC):
    def __init__(self, bedrock_runtime_client, model: str):
        self._client = bedrock_runtime_client
        self.model = model

    @abstractmethod
    def _format_body(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = 1000,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> Dict:
        ...

    @abstractmethod
    def _parse_completion(self, completion, caller_id: Optional[str] = None) -> ChatMessage:
        ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = 1000,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        # create the json body to send to the API
        body = self._format_body(messages, temperature, max_tokens, caller_id, seed, format)

        # invoke the model and get the response
        try:
            accept = "application/json"
            contentType = "application/json"
            response = self._client.invoke_model(body=body, modelId=self.model, accept=accept, contentType=contentType)
            completion = json.loads(response.get("body").read())
        except RuntimeError as err:
            raise LLMConfigurationError("Could not get response from Bedrock API") from err

        return self._parse_completion(completion, caller_id)

class MistralBedrockClient(BaseBedrockClient):
    def __init__(
        self,
        bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"]),
        model: str = "mistral.mistral-7b-instruct-v0:2",
        anthropic_version: str = "bedrock-2023-05-31",
    ):
        # only supporting claude 3
        # if "claude-3" not in model:
        #     raise LLMConfigurationError(f"Only claude-3 models are supported as of now, got {self.model}")

        super().__init__(bedrock_runtime_client, model)
        self.anthropic_version = anthropic_version

    def _format_body(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = 1000,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> Dict:
        input_msg_prompt: List = []
        system_prompts = []

        for msg in messages:
            # System prompt is a specific parameter in Claude
            if msg.role.lower() == "system":
                system_prompts.append(msg.content)
                continue

            # Only role user and assistant are allowed
            role = msg.role.lower()
            role = role if role in ["assistant", "user"] else "user"

            # Consecutive messages need to be grouped
            last_message = None if len(input_msg_prompt) == 0 else input_msg_prompt[-1]
            if last_message is not None and last_message["role"] == role:
                last_message["content"].append({"type": "text", "text": msg.content})
                continue

            input_msg_prompt.append({"role": role, "content": [{"type": "text", "text": msg.content}]})

        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": "\n".join(system_prompts),
                "messages": input_msg_prompt,
            }
        )

    def _parse_completion(self, completion, caller_id: Optional[str] = None) -> ChatMessage:
        self.logger.log_call(
            prompt_tokens=completion["usage"]["input_tokens"],
            sampled_tokens=completion["usage"]["output_tokens"],
            model=self.model,
            client_class=self.__class__.__name__,
            caller_id=caller_id,
        )

        msg = completion["content"][0]["text"]
        return ChatMessage(role="assistant", content=msg)

set_default_client(MistralBedrockClient())

# Load and split the PDF document
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
# climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# # Test the QA chain with a sample question
# question = "What are the impacts of climate change on agriculture?"
# response = climate_qa_chain.run(question)
# print(response)
def get_claude_response(prompt: str):
    # Use HumanMessage for the user's input
    messages = [HumanMessage(content=prompt)]
    
    # Send the formatted messages to the Claude model
    response = llm(messages=messages, max_tokens=500, temperature=0.7)
    return response.content

# Example usage
question = "What are the impacts of climate change on agriculture?"
response = get_claude_response(question)
print("Claude Response:", response)


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