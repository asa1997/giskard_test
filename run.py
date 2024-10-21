from transformers import AutoTokenizer, AutoModelForCausalLM
import giskard

import os

from giskard.llm.client.mistral import MistralClient

os.environ['MISTRAL_API_KEY'] = "e33RIAPiVKoYUAUjldIxmTYHIqzkYZr4H/fvYvhc"

mc = MistralClient()

giskard.llm.set_default_client(mc)
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

giskard_model = giskard.Model(
    model=model,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=["question"],
)

scan_results = giskard.scan(giskard_model)