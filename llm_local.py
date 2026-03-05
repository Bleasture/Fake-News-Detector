from llama_index.llms.llama_cpp import LlamaCPP

def load_llm():
    llm = LlamaCPP(
        model_path=r"C:\SeamRag\seamrag\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.5,
        max_new_tokens=512,
        context_window=4096,
        model_kwargs={
            "n_ctx": 4096,
            "n_threads": 8,
            "n_gpu_layers": -1,
            "n_batch": 1024,
            "main_gpu": 0,
            "verbose": False
        }
    )
    return llm

import json
import re

def analyze_article(article_text, llm):

    prompt = f"""
You are a strict fake news classification system.

Step 1: Extract the main factual claims explicitly stated in the article.
These are statements the article presents as facts.

Step 2: Classify the article as Fake, Real, or Uncertain.

Classification Rules:
- Extraordinary scientific claims without peer review → Fake
- Invented scientific terminology → Fake
- Very small sample size for extraordinary claims → Fake
- Lack of credible sources → Fake

Return ONLY valid JSON in this format:

{{
  "verdict": "Fake | Real | Uncertain",
  "confidence": number between 0-100,
  "key_claims": ["factual claim 1", "factual claim 2"],
  "explanation": "brief reasoning for classification"
}}

Important:
- key_claims must contain the actual claims made in the article.
- Do NOT include reasons for why it is fake inside key_claims.
- Reasons belong ONLY in the explanation field.
- Do not include any text outside the JSON.
- Extract at least 3–5 key factual claims if available.

Article:
{article_text}

JSON:
"""

    response = llm.complete(prompt)
    text = response.text.strip()

    # Try extracting JSON safely
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # fallback if parsing fails
    return {
        "verdict": "Parsing Error",
        "confidence": 0,
        "key_claims": [],
        "explanation": text
    }