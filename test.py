import giskard
from typing import Dict, List, Optional, Sequence
import os
import json
from abc import ABC, abstractmethod
import datasets
from giskard.llm.client import set_default_client
from giskard.llm.config import LLMConfigurationError
from giskard.llm.errors import LLMImportError
from giskard.llm.client.base import LLMClient, ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import pandas as pd
import huggingface_hub


# from .base import ChatMessage

try:
    import boto3  # noqa: F401
except ImportError as err:
    raise LLMImportError(
        flavor="llm", msg="To use Bedrock models, please install the `boto3` package with `pip install boto3`"
    ) from err


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

def _select_best_dataset_split(split_names):
        """Get the best split for testing.

        Selects the split `test` if available, otherwise `validation`, and as a last resort `train`.
        If there is only one split, we return that split.
        """
        # If only one split is available, we just use that one.
        if len(split_names) == 1:
            return split_names[0]

        # Otherwise iterate based on the preferred prefixes.
        for prefix in ["test", "valid", "train"]:
            try:
                return next(x for x in split_names if x.startswith(prefix))
            except StopIteration:
                pass

        return None

def load_dataset(
        dataset_id, dataset_config=None, dataset_split=None, model_id=None
    ):
        """Load a dataset from the HuggingFace Hub."""
        logger.debug(
            f"Trying to load dataset `{dataset_id}` (config = `{dataset_config}`, split = `{dataset_split}`)."
        )
        try:
            # we do not set the split here
            # because we want to be able to select the best split later with preprocessing
            hf_dataset = datasets.load_dataset(dataset_id, name=dataset_config)

            if isinstance(hf_dataset, datasets.Dataset):
                logger.debug(f"Loaded dataset with {hf_dataset.size_in_bytes} bytes")
            else:
                logger.debug("Loaded dataset is a DatasetDict")

            if dataset_split is None:
                dataset_split = self._select_best_dataset_split(list(hf_dataset.keys()))
                logger.info(
                    f"No split provided, automatically selected split = `{dataset_split}`)."
                )
                hf_dataset = hf_dataset[dataset_split]

            return hf_dataset
        except ValueError as err:
            msg = (
                f"Could not load dataset `{dataset_id}` with config `{dataset_config}`."
            )
            raise DatasetError(msg) from err

def _flatten_hf_dataset(hf_dataset, data_split=None):
        """
        Flatten the dataset to a pandas dataframe
        """
        flat_dataset = pd.DataFrame()
        if isinstance(hf_dataset, datasets.DatasetDict):
            keys = list(hf_dataset.keys())
            for k in keys:
                if data_split is not None and k == data_split:
                    # Match the data split
                    flat_dataset = hf_dataset[k]
                    break

                # Otherwise infer one data split
                if k.startswith("train"):
                    continue
                elif k.startswith(data_split):
                    # TODO: only support one split for now
                    # Maybe we can merge all the datasets into one
                    flat_dataset = hf_dataset[k]
                    break
                else:
                    flat_dataset = hf_dataset[k]

            # If there are only train datasets
            if isinstance(flat_dataset, pd.DataFrame) and flat_dataset.empty:
                flat_dataset = hf_dataset[keys[0]]

        return flat_dataset

def _find_dataset_id_from_model(model_id):
    """Find the dataset ID from the model metadata."""
    model_card = huggingface_hub.model_info(model_id).cardData

    if "datasets" not in model_card:
        msg = f"Could not find dataset for model `{model_id}`."
        raise DatasetError(msg)

    # Take the first one
    dataset_id = model_card["datasets"][0]
    return dataset_id


# model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
giskard_model = giskard.Model(
    model=model,
    model_type="classification",
    name="DistilBERT SST-2",
    data_preprocessing_function=lambda df: tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ),
    feature_names=["text"],
    classification_labels=["negative", "positive"],
    batch_size=32,  # set the batch size here to speed up inference on GPU
)

dataset_id = _find_dataset_id_from_model('distilbert-base-uncased-finetuned-sst-2-english')

hf_dataset = load_dataset(dataset_id)

gsk_dataset = _flatten_hf_dataset(hf_dataset, '')


scan_results = giskard.scan(giskard_model, gsk_dataset)

print(scan_results)