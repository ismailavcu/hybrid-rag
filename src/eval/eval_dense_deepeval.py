# https://deepeval.com/guides/guides-rag-evaluation

# for openai/huggingface models, load .env
from dotenv import load_dotenv
import os

load_dotenv()

#openai_key = os.getenv('OPENAI_API_KEY')
try:
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
except:
    hf_key = None

from src.retrieval.dense import  DenseRetriever
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from src.eval.ground_truths import questions, ground_truths
from src.ingestion.chunker import chunk_text
from src.ingestion.pdf_loader import load_pdf
from src.rag.llm import llm

# imports below are to create CustomLlama3_1_8B class for metric measure
from deepeval.models import DeepEvalBaseLLM
import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama




# local ollama llm for test case metric.measure
# https://deepeval.com/guides/guides-using-custom-llms#creating-a-custom-llm
################   I have used CustomDolphin3_8B instead of CustomLlama3_1B !!  ##################################
class CustomLlama3_1B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
            token = hf_key
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", # "meta-llama/Meta-Llama-3-8B-Instruct",
            token = hf_key
        )
        self.model = model_4bit
        self.tokenizer = tokenizer
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pipeline(prompt)
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    def get_model_name(self):
        return "Llama-3.2 1B" # "Llama-3 8B"
    



class CustomDolphin3_8B(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "dolphin3:8b"

    def load_model(self):
        # Ollama runs as a service, so nothing to load here
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluation model. Always return valid JSON. Return ONLY valid JSON. No explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0  # deterministic -> better for eval
            }
        )

        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        # simple async wrapper
        return self.generate(prompt)

    def get_model_name(self):
        return "dolphin3-8b"
    
#custom_llm_for_metrics = CustomLlama3_1B() # this may take too long
custom_llm_for_metrics = CustomDolphin3_8B()

contextual_precision = ContextualPrecisionMetric(model=custom_llm_for_metrics)
contextual_recall = ContextualRecallMetric(model=custom_llm_for_metrics)


if __name__ == "__main__":
    # python -m src.eval.eval_dense_deepeval

    # First, embed docs
    file_path = r"data/raw/History of BMW - Wikipedia.pdf"
    texts = load_pdf(file_path)
    chunks = chunk_text(texts)

    retriever = DenseRetriever(chunks)

    test_cases = []

    # Build test cases
    for question, gt in zip(questions, ground_truths):

        results = retriever.search(question, top_k=5)

        contexts = [doc for doc, _ in results]
        context_wo_id = [i[0] for i in contexts] #contexts without distances
        answer = llm(question, context_wo_id)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=gt,
            retrieval_context=context_wo_id,
        )

        test_cases.append(test_case)

    
    # Metrics
    metrics = [
        contextual_precision,
        contextual_recall,
    ]

    # Run evaluation
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i} ---")
        print(f"Q: {test_case.input}")

        for metric in metrics:
            try:
                metric.measure(test_case)
                print(f"{metric.__class__.__name__}: {metric.score:.4f}")
            except Exception as e:
                print(f"{metric.__class__.__name__}: FAILED ({e})")


"""Expected Outout is like:

--- Test Case 0 ---
Q: When was BMW officially founded?
ContextualPrecisionMetric: 1.0000
ContextualRecallMetric: 0.5000

--- Test Case 1 ---
Q: What was BMW originally called?
ContextualPrecisionMetric: 1.0000
ContextualRecallMetric: 0.5000

--- Test Case 2 ---
Q: What was BMW’s first product?
ContextualPrecisionMetric: 1.0000
ContextualRecallMetric: 0.4000

--- Test Case 3 ---
Q: What did BMW produce after World War I to survive?
ContextualPrecisionMetric: 1.0000
ContextualRecallMetric: 0.8000

...

"""

        