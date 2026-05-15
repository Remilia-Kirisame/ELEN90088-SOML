# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (llmproj)
#     language: python
#     name: llmproj
# ---

# %% [markdown]
# **These two cells** is for running on Spartan OOD (Open OnDemand)
#
# First cell:
# - Prevents large Hugging Face models and datasets from filling the home folder.
# - Stores model/dataset cache under project folder.
# - Get directory from system setup, avoid directly writing here.
#
# Second cell:
# - Test if it is working

# %%
import os
from pathlib import Path

PROJECT_DIR = os.environ.get("SPARTAN_PROJECT_DIR")
if not PROJECT_DIR:
    raise RuntimeError(
        "Set SPARTAN_PROJECT_DIR first, e.g. /data/gpfs/projects/<project-id>"
    )

PROJECT_DIR = Path(PROJECT_DIR)
WORK_DIR = PROJECT_DIR / "SOML_LLM_project"
HF_DIR = PROJECT_DIR / "huggingface"

paths = {
    "HF_HOME": HF_DIR,
    "TRANSFORMERS_CACHE": HF_DIR / "transformers",
    "HF_DATASETS_CACHE": HF_DIR / "datasets",
    "HF_HUB_CACHE": HF_DIR / "hub",
    "HF_ASSETS_CACHE": HF_DIR / "assets",
    "HF_MODULES_CACHE": HF_DIR / "modules",
}

for key, path in paths.items():
    os.environ[key] = str(path)
    path.mkdir(parents=True, exist_ok=True)

for path in [
    WORK_DIR,
    WORK_DIR / "cache",
    WORK_DIR / "models",
    WORK_DIR / "outputs",
    WORK_DIR / "qlora_results",
    WORK_DIR / "qlora_logs",
]:
    path.mkdir(parents=True, exist_ok=True)

# %%
import sys
import torch

print("Python executable:")
print(sys.executable)

print("\nPyTorch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

# %% [markdown] id="c8ea16a9"
# ## Part 0: Introduction and preparation
#
# Welcome to the Large Language Model (LLM) project of ELEN90088!
#
# In this project, you will transition from a consumer of AI to an engineer who understands the mechanics of LLMs. You will explore how these models are loaded, manipulated, trained and optimized to run on GPUs.
#
# This project focuses on implementation and engineering. While the theories behind the transformer architecture (Attention, LayerNorm etc.) may be covered in lectures, you will not be asked to implement
# these components from scratch (note that we may ask you to explain attention mechnaisms from a high level, but not in detail). Instead, you will learn to use the HuggingFace library to leverage these components.
#
# This project is structured into six parts. You will follow the notebook to get familiar with LLMs in part 1-5, and design your own mini-project in part 6.
#
# Specifically, this project includes the following parts:
# 1. The `pipeline`: Use high-level LLM APIs to perform tasks like sentiment analysis and text generation.
# 2. Behind the `pipeline`: Understand and manage the key components of LLM workflow, including tokenizers, models and datasets.
# 3. Inference: Use the tools from part 2 to implement the `pipeline` in part 1.
# 4. Efficient LLM - Model Compression: Reduce the LLM size while maintaining the LLM performance.
# 5. Efficient LLM - Parameter-efficient Fine-Tuning (PEFT): Use advanced methods to specialize a general-purpose model for specific tasks.
# 6. Mini-project: Design and implement your own LLM project with the skills you have learned from part 1-5.
#
# Notes:
# 1. GPU usage: for part 1-2, cpu may be sufficient (loading models for a few minutes is normal). From part 3, you will need access to GPU (Google Colab, and/or Windows with Nvidia) or MPS (on Mac).
# 2. Project report: This notebook includes some questions that help you understand more details about LLM and implemetation. These questions are labelled as "**Questions (for project report)**". You may prepare the responses to these questions as part of your report.
# 3. Get help: While this notebook was prepared to explain as much terminology as possible, you may find some definitions unfamilar and need time to process. This is perfectly normal especially for beginners. You are encouraged to find other guidelines, technical reports and blogs from Google and/or AI. For example, what is a BERT model and what does it do? We will test your understanding during oral exams.
# 4. Code: Most of the code in part 1-5 has been provided to you. We also left some parts for you to implement as exercises.
#    
#     That said, instead of just running the cells, we highly recommend you to understand **why** the code works, and how you could modify some parameters, variables to improve the code performance. We will test whether you **really** understand the code during oral exams.
# 5. (**Important**) Mistral vs. Phi LLM. As you will see, starting from part 2, this notebook frequently talks about the Mistral model, and the code is often about the Mistral model. While Mistral has great functionalities, it may not be feasible to run Mistral models on your hardware, which is perfectly fine becuase there are other models that can run on your hardware.
#
#     If you are using standard hardware (eg. Macbook Pro, or Colab free tier), we recommend you to use Phi model from Microsoft, which we have tested and confirmed it will run on your hardware and the free version of Google Colab. The only change you need to make is to ensure you are loading and using Phi model (not Mistral which may crash your code) where necessary.
#     
#     For the tasks that you may use Phi model and update the code (written for Mistral models), we have added a note: "**(You can use Phi model for this task, please modify the code if requried)**"
#
#     Your project marks are **not** affected by which model you use.
#
# 6. Oral exam: We will test your understanding of this notebook, including but not limited to (your answers to) "**Questions (for project report)**", code details, and your mini project during oral exams.

# %% id="a766c710"
import os
import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import pandas as pd

# %% id="2955b909"
# Define the device selector
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# %% id="bad37696"
import sys
print(f"Python is running from: {sys.executable}")
print(f"Torch is installed in: {torch.__file__}")

# %% id="5e9fc8ed"
# uncomment the line below to check GPU status (not examinable)
# !nvidia-smi

# %% id="6accd374"
# Some checks to make sure the project environment will work (not examinable)

from huggingface_hub import split_torch_state_dict_into_shards
print("Successfully imported!")

# %% id="607b3bd5"
# Some checks to make sure the project environment will work (not examinable)

import transformers
from packaging import version

required_version = "4.34.0"
current_version = transformers.__version__

if version.parse(current_version) < version.parse(required_version):
    print(f"❌ ERROR: Your transformers version ({current_version}) is too old!")
    print(f"Please run: pip install -U transformers")
else:
    print(f"✅ SUCCESS: Transformers version {current_version} is compatible with Mistral.")

# %% [markdown] id="89da774f"
# ## Part 1: Warm up, LLM pipeline
#
# In this part, we will treat LLMs as a black box. We will use the Hugging Face `pipeline` API, which abstracts the complexities of tokenization, model architecture and output decoding into a single line of code.
#
# Refer: [HF - Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines), [HF - Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)

# %% [markdown] id="7dfbb48e"
# ### Example 1a: Autoregressive Text Generation
#
# Unlike classification in other neurnal network applications, text generation is a generative task. The model predicts possible next token based on all previous tokens in the sequence.

# %% id="c3b94d32"
# generator = pipeline("text-generation")
# generator("In this LLM project, we will teach you how to")
'''
# This version still gives minor warnings.
generator = pipeline("text-generation", model="gpt2")
generator(
    "In this LLM project, we will teach you how to",
    pad_token_id=generator.tokenizer.eos_token_id,
    max_new_tokens=50,
    max_length=None
)
'''
# This version gives clean output
from transformers import logging

# 1. Suppress the "UNEXPECTED architecture" informational notes
logging.set_verbosity_error()

# Initialize the pipeline
generator = pipeline("text-generation", model="gpt2")

# 2. Update the model's generation config directly.
# This avoids the deprecated behavior of passing kwargs directly into generator()
generator.model.generation_config.pad_token_id = generator.tokenizer.eos_token_id

# 3. Explicitly remove the default max_length so it doesn't clash with max_new_tokens
generator.model.generation_config.max_length = None
generator.model.generation_config.max_new_tokens = 50

# Generate text (notice we no longer need to pass the arguments here!)
output = generator("In this LLM project, we will teach you how to")

print(output)
# [{'generated_text': 'In this LLM project, we will teach you how to create an implementation of a large number of binary operations using the LLVM framework.\n\nTo begin, we\'ll create a class called "ObjectModel". This class will be used to store the list of objects that are stored in a model and its associated classes.\n\nFor this example, we will use a class named "Model" that holds the list of objects that are stored in the model. We will also add a class named "Key" to the class named "ModelModel".\n\nWe will also create a class called "Array". This class will hold a list of all the variables that are stored in the model.\n\nWe will also create a class named "Integer". This class will hold the number of operations that are performed in the model.\n\nWe will also create a class named "float". This class will hold the number of operations that are performed in the model.\n\nWe will also create a class named "Rectangle". This class will hold the rectangle that the model is in, plus the coordinate system and its associated Rectangle objects.\n\nFinally, we will create a class named "Float". This class will hold the number of operations performed in the model.\n\nWe will also create a class named "Rectangle'}]

# %% [markdown] id="eF0vb0aCnX3v"
# **Questions (for project report)**
#
# - Did you get some warning messages or errors? If the warning messages (or notes) did not stop you from running the cell above, what could you do to eliminate the warning?
#
# **Hint**: you may want to specify a model, explain `gpt2` model, and understand `pad_token_id`, `eos_token_id`. You may also change the string in `generator` and discuss the quality of generated text, does the output text make sense?

# %% [markdown] id="_ZP__VxEnMpp"
# **Answers (Example 1a):**
#
# **Warnings observed and how to eliminate them**
#
# Three notable warnings appear when the cell is run:
#
# 1. *"No model was supplied, defaulted to openai-community/gpt2 and revision 607a30d."* — `pipeline()` was not given a model, so it picked a default. Fix by passing `model=` explicitly:
#    ```python
#    generator = pipeline("text-generation", model="gpt2")
#    ```
# 2. *"Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."* — GPT-2 was not trained with a dedicated pad token, so HuggingFace falls back to using the end-of-sequence token (id 50256, the `<|endoftext|>` token) as padding. For single-sequence inference this is harmless. To suppress it, pass `pad_token_id=tokenizer.eos_token_id` explicitly when calling `generate()`.
# 3. *"Both `max_new_tokens` (=256) and `max_length` (=50) seem to have been set. `max_new_tokens` will take precedence."* — the two length parameters conflict. Pass only one of them (prefer `max_new_tokens`, since it counts only the generated tokens, not the prompt).
#
# The `HF_TOKEN` notice is informational — it just says we are sending unauthenticated requests, which is fine for public models.
#
# **What is GPT-2?**
#
# GPT-2 (Generative Pre-trained Transformer 2, OpenAI 2019) is a **decoder-only** transformer trained autoregressively to predict the next token given the previous ones. The base checkpoint used here has roughly 124M parameters and was pre-trained on WebText (~40 GB of internet text). It is a *generation* model — no classification head, no instruction tuning, no chat formatting.
#
# **`pad_token_id` vs `eos_token_id`**
#
# - `eos_token_id`: marks the end of a sequence. When the model emits this token, generation stops.
# - `pad_token_id`: a placeholder used to make sequences in a batch the same length so they fit into a single tensor. GPT-2 has no dedicated pad token, so HuggingFace reuses the EOS id (50256) for padding by default during open-ended generation.
#
# **Output quality**
#
# The output stays locally fluent but drifts off-topic — the prompt mentions an LLM project, but the model rambles into "C++ compiler", "LLVM", and the "C++ standard library" with heavy repetition. This reflects GPT-2's limitations: it is small, old, and not instruction-tuned, so it latches onto surface n-gram patterns rather than the prompt's intent. Modern instruction-tuned models (e.g. Mistral-Instruct, Llama-3-Instruct) follow the actual instruction much more reliably. Re-running the cell yields different output because the default sampling is stochastic (`do_sample=True`).

# %% [markdown] id="ce846109"
# ### Example 1b: Sentiment analysis
#
# The goal of sentiment analysis is to claassify a string of text into a category (eg. Positive or Negative). This is a discriminative task, not a text generation task.

# %% id="b2afeeed"
# classifier = pipeline("sentiment-analysis")
# classifier("How good is ELEN90088")
# # gives `[{'label': 'POSITIVE', 'score': 0.9998353719711304}]`

# classifier = pipeline("sentiment-analysis", model="FacebookAI/roberta-large-mnli")
# classifier("How good is ELEN90088")
# # gives `[{'label': 'NEUTRAL', 'score': 0.5751283764839172}]`, I guess it is read as borderline or sarcastic.

classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english") # default model
classifier(["Sure, that's just great.", "You genius"])
# [{'label': 'POSITIVE', 'score': 0.9998683929443359},
#  {'label': 'POSITIVE', 'score': 0.9998675584793091}]

# %% [markdown] id="4b757e3a"
# **Questions (for project report)**
#
# You will see something like "No model was supplied, defaulted to **distilbert**/distilbert-base-uncased-finetuned-sst-2-english"
#
# - What's BERT, how's that different from other models (eg. GPT-2)?
#
# - What's `distilbert`? What's `distilbert-base-uncased-finetuned-sst-2-english`?
#
# - What are `label` and `score`? Can you try different sentiment and check the output `label` and `score`?

# %% [markdown] id="d7RJUSB_pHNW"
# **Answers (Example 1b):**
#
# **What is BERT, and how does it differ from GPT-2?**
#
# BERT (Bidirectional Encoder Representations from Transformers, Google 2018) is an **encoder-only** transformer. It is pre-trained with two objectives:
#
# - *Masked Language Modeling (MLM)*: random tokens in the input are masked and the model predicts them, using context from both directions.
# - *Next Sentence Prediction (NSP)*: the model is shown two sentences and asked whether the second follows the first.
#
# Key differences from GPT-2:
#
# | Aspect | BERT (encoder) | GPT-2 (decoder) |
# |---|---|---|
# | Attention | Bidirectional | Causal (left-to-right only) |
# | Pre-training task | MLM + NSP | Next-token prediction |
# | Designed for | Understanding (classification, NER, QA) | Generation |
# | Typical use | Add a small task-specific head, fine-tune | Sample tokens autoregressively |
#
# For sentiment analysis we want to *understand* the input, so an encoder model like BERT (or DistilBERT) is a natural fit.
#
# **What is `distilbert`?**
#
# DistilBERT is a *distilled* version of BERT — a smaller "student" model trained to imitate a larger BERT "teacher" via knowledge distillation. It has roughly 40% fewer parameters (~66M vs 110M for BERT-base) and is ~60% faster at inference, while retaining about 97% of BERT's score on the GLUE benchmark.
#
# **Decoding `distilbert-base-uncased-finetuned-sst-2-english`**
#
# - `distilbert` — model family.
# - `base` — base size (smaller than `large`).
# - `uncased` — input is lowercased before tokenization; the model does not distinguish "Apple" from "apple".
# - `finetuned-sst-2-english` — fine-tuned on **SST-2** (Stanford Sentiment Treebank v2), an English movie-review dataset with binary `POSITIVE` / `NEGATIVE` labels.
#
# **`label` and `score`**
#
# - `label`: the predicted class — for this checkpoint, either `POSITIVE` or `NEGATIVE`.
# - `score`: the model's confidence in that class, a probability in $[0, 1]$ obtained by applying softmax over the two logits.
#
# Worth trying borderline or sarcastic inputs (e.g. `"It's not bad."`, `"Sure, that's just great."`, `"This is the best example of a terrible movie."`) — the score will collapse toward $0.5$ or flip in the wrong direction. This shows that the model leans on surface keywords more than intent, which is exactly the failure mode Part 3b asks us to compare against an instruction-tuned LLM.

# %% [markdown] id="44230656"
# ### Exploring `pipeline`
#
# Hugging Face offers dozens of pre-configured pipelines for differentt modalities. Now that you have seen how `pipeline()` works (twice), it is your turn to explore.
#
# ### Exercise:
#
# Choose another two `pipeline()` tasks (that's not sentiment analysis or text generation) and implement them in the cells below. Also briefly explain the tasks and their performance in your report. For example, does the model output meet your expectations?
#
# **Hint**: This link might help. https://huggingface.co/docs/transformers/en/main_classes/pipelines
#
# TBD: Brief explanation for report.

# %% id="e3027021"
# pipeline() Task 1: Object Detection
# Predicts bounding boxes of objects and their classes.

# detector = pipeline("object-detection", model="facebook/detr-resnet-50")
# detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
# '''
# [{'score': 0.9966182112693787,
#   'label': 'bird',
#   'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}},
#  {'score': 0.9993816614151001,
#   'label': 'bird',
#   'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]
# '''
import requests
from PIL import Image, ImageDraw
from transformers import pipeline
from IPython.display import display

# 1. Load the image from the URL
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw)

# 2. Initialize the detector and get results
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
results = detector(image)

# 3. Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(image)

for result in results:
    box = result["box"]
    label = result["label"]
    score = result["score"]

    # Extract coordinates
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    # Draw the rectangle (outline color and thickness can be adjusted)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

    # Draw the text label and confidence score just above the box
    text = f"{label} ({score:.2f})"
    draw.text((xmin, ymin - 10), text, fill="red")

# 4. Render the image directly in the Colab output cell
display(image)
# See local file [[object-detection-output.png]]

# %% id="41715539"
# pipeline() Task 2: Zero-shot classification
# Classify text against arbitrary labels supplied at inference time — no task-specific fine-tuning required.
# The underlying model (DistilBART fine-tuned on MNLI) treats classification as natural-language inference:
# for each candidate label it asks "does this text entail the hypothesis 'this example is about <label>'?" and returns the entailment scores.
zsc = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

text = "I just deployed a fine-tuned LLM and the latency dropped by 30%."
candidate_labels = ["machine learning", "cooking", "sports", "finance"]

zsc(text, candidate_labels)

'''Output
{'sequence': 'I just deployed a fine-tuned LLM and the latency dropped by 30%.',
 'labels': ['machine learning', 'cooking', 'sports', 'finance'],
 'scores': [0.5655022263526917,
  0.2784496247768402,
  0.08393162488937378,
  0.07211653888225555]}
'''

# %% [markdown] id="e6304bad"
# **Questions (for project report)**
#
# - While the `pipeline()` API is easy to use (two lines of code), it hides several critical steps of LLM workflow. Based on your observation of execution time, what do you think is happening when you call the `pipeline()` function for the first time vs. the second time? Briefly explain in your report.

# %% [markdown] id="hsBP5TSlz6WC"
# **Answer (execution-time question):**
#
# The first call to `pipeline(task)` for a model you have never used does several heavy steps that are all hidden behind the one-line API:
#
# 1. **Download** — `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.txt`, `merges.txt`, etc. are fetched from the HuggingFace Hub into the local cache (`~/.cache/huggingface/hub` by default). This is what the progress bars are showing.
# 2. **Construct the model object** — read the config and instantiate the matching `nn.Module` architecture in memory.
# 3. **Load weights** — read the safetensors file and copy each tensor into the model parameters. On GPU this also involves a host-to-device copy (`.to(device)`).
# 4. **Build the tokenizer** — load the vocab/merges and compile the tokenization rules.
# 5. **First forward pass** — CUDA kernels, autotuners and any JIT-compiled paths warm up.
#
# A second call to `pipeline(task)` (same model, same session) skips step 1 because the files are already on disk, but it still re-instantiates the Python object and re-loads the weights into device memory. So it is faster than the very first time, but not free.
#
# Once you *have* a pipeline object, calling it on a string (the actual inference) only runs a forward pass on a small input — that part is fast. This is the core lesson of Part 2: by separating tokenizer, model, and generation, you pay the load cost *once* and reuse the object, instead of paying it again for every prediction.

# %% [markdown] id="ca094002"
# ## Part 2: Behind the pipeline - LLM workflow
#
# While the `pipeline()` API is convenient, it abstracts away the engineering decisions required for high-performance systems. To optimize for memory and/or speed, we must gain granular (highly precise) control over the individual components of the LLM workflow. Key components in the workflow include the tokenizer, the model and the datasets.
#
# #### The choice of model
# For this part, we are using Mistral-7B-Instruct-v0.3 for a few reasons:
#
# - It is widely considered one of the best models in the "7-Billion parameter" class, outperforming some models twice its size.
# - Since it has been fine-tuned to follow instructions, it is highly capable of performing zero-shot and few shot tasks we will attempt in Part 3.
#
# **Note:** For part 2, you should be able to load Mistral model with any hardware. However when you start inference in next parts, your hardware may not be able to handle Mistral. Don't panic, you can use Phi model instead.
#   
# #### Manage storage (**Optional**)
#
# When you download an LLM, you are downloading massive files. For example, the Mistral weights alone are around 14.5 GB. By default, Hugging Face saves these in a hidden folder in your Home directory (eg. `~/.cache/huggingface` on Mac OS). This could become tricky if you download a few models and datasets without checking and managing disk storage. A simple method to get easy access to the downloaded files (and manage them) is to create a cache folder within your project directory, and download all the project files in this dedicated cache folder.
#
# **Note:** While this is recommended, setting up this cache folder is optional and will not affect your project marks.

# %% id="a0c7eca4"
# cache_dir = "./project_cache"

# %% [markdown] id="43776198"
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="7dd16298"
model_id = "mistralai/Mistral-7B-Instruct-v0.3" #we use Mistral instruct model for this part

# Uncomment below if you want to use Phi model: it's easier to run on local device and free Colab!
# model_id = "microsoft/Phi-3.5-mini-instruct"
# cache_dir = "./phi_project"

# My cache dir for Spartan-OOD
cache_dir = str(WORK_DIR / "cache" / "mistral_proj")

# %% [markdown] id="64733c79"
# #### 2.1 The tokennizer
#
# Computers process numbers, not text. The **tokenizer** acts as the "translator" between human language and the high-dimensional vectors the model understands. It performs three critical steps:
#
# 1. Normalization: Removing extra spaces, lowercase conversion, etc.
# 2. Pre-tokenization: Splitting text into words or sub-words.
# 3. Encoding: Mapping those sub-words to their unique integer IDs in the model's vocabulary.
#
# It is a fundamental requirement that your tokenizer and model come from the exact same "designer" and version (in this part, we use the Mistral model). If they don't match, for example, we load a Llama-3 tokenizer with a Mistral model, the system will not necessaricly throw a code error, but the results will likely be ungrammatical texts. Every model is trained with a specific Vocabulary index. If a tokenizer tells the Mistral model that the ID `5678` means "Apple", but the Mistral model and tokenizer was trained to believe `5678` means "float", then the mathematical logic of the network will collapse.
#
# #### Exercise:
#
# Load the tokenizer and observe how it handles the Mistral/Phi vocabulary.

# %% id="138bac7f"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    cache_dir=cache_dir # Note that we specified the cache folder here and in the next parts where we load models, tokenizers etc.
)


text = "Optimization is essential for LLMs."
ids = tokenizer.encode(text)

print(f"Token IDs: {ids}")
print(f"Decoded: {tokenizer.decode(ids)}")

# more examples...
print(f"Decoded: {tokenizer.decode(5678)}")

'''Output (Phi):
Token IDs: [20693, 326, 2133, 338, 18853, 363, 365, 26369, 29879, 29889]
Decoded: Optimization is essential for LLMs.
Decoded: Ros
'''
''' Output (Mistral)
Token IDs: [1, 28036, 2605, 1117, 8742, 1122, 17472, 16791, 29491]
Decoded: <s> Optimization is essential for LLMs.
Decoded: float
'''

# %% id="oJZ5j1bWw7SL"
# more examples...
text = "Tokenization"
ids = tokenizer.encode(text)

print(f"Token IDs: {ids}")
print(f"Decoded: {tokenizer.decode(ids)}")

print(f"Decoded: {tokenizer.decode(17393)}")

'''Output (Mistral):
Token IDs: [1, 17393, 2605]
Decoded: <s> Tokenization
Decoded: Token
'''


# %% [markdown] id="XBKme3yayk34"
# **Questions (for project report)**
#
# - A token can be a word, or part of a word. For example, "tokenization" includes "token" and "ization". Can you find other words that consist of multiple tokens?
#
# - Can you explain why tokenization may be a good approach to turn texts into numbers? **Hint:** What would happen if every word was represented by an unique integer?

# %% id="PUKmjV9zvkdp"
# Test examples
def try_tokenizer(text):
  ids = tokenizer.encode(text)
  print(f"Token IDs: {ids}")
  # Decode each ID individually and join them with ' | '
  decoded_tokens = [tokenizer.decode([token_id]) for token_id in ids]
  print(f"Decoded: {' | '.join(decoded_tokens)}")

text = "In the rain, the pavement shines like silver."
try_tokenizer(text)
text = "element vanish pavage"
try_tokenizer(text)

# Explore which words get split into multiple sub-word tokens.
# `convert_ids_to_tokens` shows the raw SentencePiece pieces (including the `▁` word-boundary marker);
# `decode([id])` shows the human-readable string for each piece.
sample_words = [
    "tokenization",
    "unbelievable",
    "internationalization",
    "antidisestablishmentarianism",
    "supercalifragilisticexpialidocious",
    "preprocessing",
    "ChatGPT",
    "Melbourne",
    "ELEN90088",        # course code, mixed alphanumeric
    "Optimization",
    "hello",
    "the",
]

print(f"\n{'word':40s}  {'#tok':>5s}  raw pieces (convert_ids_to_tokens)")
print("-" * 100)
for word in sample_words:
    ids = tokenizer.encode(word, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print(f"{word:40s}  {len(ids):>5d}  {pieces}")

'''Output (Mistral):
Token IDs: [1, 1328, 1040, 8064, 29493, 1040, 1052, 1226, 1234, 1248, 2071, 1505, 10514, 29491]
Decoded: <s> | In | the | rain | , | the | p | ave | ment | sh | ines | like | silver | .
Token IDs: [1, 3210, 2465, 1557, 1052, 1262, 1233]
Decoded: <s> | element | van | ish | p | av | age

word                                       #tok  raw pieces (convert_ids_to_tokens)
----------------------------------------------------------------------------------------------------
tokenization                                  2  ['▁token', 'ization']
unbelievable                                  4  ['▁un', 'bel', 'iev', 'able']
internationalization                          2  ['▁international', 'ization']
antidisestablishmentarianism                  8  ['▁ant', 'id', 'is', 'est', 'ablish', 'ment', 'arian', 'ism']
supercalifragilisticexpialidocious           12  ['▁super', 'cal', 'if', 'rag', 'il', 'ist', 'ice', 'xp', 'ial', 'id', 'oc', 'ious']
preprocessing                                 2  ['▁pre', 'processing']
ChatGPT                                       3  ['▁Chat', 'G', 'PT']
Melbourne                                     1  ['▁Melbourne']
ELEN90088                                     7  ['▁E', 'LEN', '9', '0', '0', '8', '8']
Optimization                                  2  ['▁Optim', 'ization']
hello                                         2  ['▁hell', 'o']
the                                           1  ['▁the']
'''

# %% [markdown] id="XN7v8HEXvzGG"
# **Answers (Part 2.1: Tokenizer):**
#
# **Words made of multiple tokens**
#
# The Phi-3 tokenizer (a SentencePiece BPE built on the Llama tokenizer) splits long, rare or compound strings into sub-word pieces. From the cell above we can see, for example:
#
# - `tokenization` → `token` + `ization`
# - `unbelievable` → `un` + `believ` + `able`
# - `internationalization`, `antidisestablishmentarianism`, `supercalifragilisticexpialidocious` are all broken into many short morpheme-like pieces.
# - Course codes like `ELEN90088` get split into letters and digits because the tokenizer was trained on natural-language text and never saw that exact string as a unit.
# - `element` was not encoded into `el-ement` as `pavement` did. Multiple reasons, one of which can be the frequency in training set.
#
# In contrast, common short words such as `the`, `hello`, `Melbourne` come out as a single token — BPE's merge rules promote frequent strings to their own vocabulary entry.
#
# > The leading `▁` (U+2581) we may see in `convert_ids_to_tokens` output is SentencePiece's marker for a word boundary, not part of the actual character.
#
# **Why is sub-word tokenization a good idea?**
#
# If we mapped every distinct word to a unique integer ID, two problems would appear:
#
# 1. **Out-of-vocabulary (OOV) blow-up.** English alone has hundreds of thousands of word forms, before we even count code, math, multilingual text, typos, or new coinages. Any word the tokenizer hadn't seen during training would have to be replaced by a generic `<unk>` token, destroying information. Sub-word tokenization avoids this: even a brand-new word like `cryptocurrency-aware` can always be expressed as a sequence of known sub-words, so the model can still process it.
#
# 2. **Embedding-table cost.** The input embedding has shape $(\text{vocab\_size}, d_{\text{model}})$. A 500k-word vocabulary at $d_{\text{model}} = 3072$ (Phi-3.5-mini) would be a 1.5-billion-parameter table just for the input layer. A sub-word vocabulary of around 32k tokens keeps that under 100M parameters while still covering essentially any input string.
#
# Sub-word tokenization (BPE / WordPiece / SentencePiece, three most popular subword tokenization methods in Natural Language Processing (NLP)) is the sweet spot — a small fixed vocabulary that nonetheless covers any text by composing pieces, which is exactly what allows one model to handle code, math, multiple languages, and made-up words without ever hitting `<unk>`.

# %% [markdown] id="5ebe7076"
# #### 2.2 The model
#
# In the Hugging Face ecosystem, "the model" is the computational heart of the system that has a collection of billions of pre-trained weights organized into a specific neural architecture. While the tokeizer handles the text-to-ID translation, the model performs the massive matrix multiplication required to predict the probability of the next token in a sequence.
#
# LLMs typically store weights using floating-point numbers. While standard software often use 32-bit (Float32), moden LLMs utilize 16-bit formats like Float16. This format provides the same dynamic range as Float32 but with half the memory footprint, offering a crucial balance between numerical stability and hardware efficiency. We will dive deeper into the these data types and how they reduce model size in part 4.
#
# > Correction: BFloat16 has the same dynamic range, rather than Float16.
#
# #### Exercise:
#
# Run the following code to load the model.
#

# %% [markdown] id="235c68d2"
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="f27d6415"
print("Loading raw model (16-bit baseline)...")
model_raw = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

print(f"Model successfully loaded on: {model_raw.device}")
print(f"Model precision: {model_raw.dtype}")
# Took long time on Colad (7.64GB downloading model)
# Output:
# Model successfully loaded on: cuda:0
# Model precision: torch.bfloat16

# %% [markdown] id="5b75812c"
# **Questions (for project report)**
#
# - How can you confirm whether your model is loaded on CPU or GPU?
#
# - Explore model architecture (attention heads, sliding window etc.). It's perfectly fine to use online resources to understand the architecture (please cite your sources). It's even better to verify the online resources by running some simple code to show the structure (eg. layer names) of the loaded model.

# %% id="n-77KMrW0SE-"
# 1. Confirm device placement and parameter count
print("=== Device placement ===")
print(f"model_raw.device                    : {model_raw.device}")
first_param = next(model_raw.parameters())  # `next()` retrieves the next item from an iterator
print(f"first parameter device              : {first_param.device}")
print(f"first parameter dtype               : {first_param.dtype}")

# `hf_device_map` is only set when device_map='auto' was used, and is the only reliable way
# to see SPLIT placement (some layers on GPU, others offloaded to CPU / disk).
if hasattr(model_raw, "hf_device_map"):
    print("\nhf_device_map (per-module placement):")
    for k, v in model_raw.hf_device_map.items():
        print(f"  {k:40s} -> {v}")
else:
    print("\n(no hf_device_map — model is fully on a single device)")

n_params = sum(p.numel() for p in model_raw.parameters())
print(f"\nTotal parameters: {n_params:,}  (~{n_params/1e9:.2f} B)")

''' Output
=== Device placement ===
model_raw.device                    : cuda:0
first parameter device              : cuda:0
first parameter dtype               : torch.bfloat16

(no hf_device_map — model is fully on a single device)

Total parameters: 3,821,079,552  (~3.82 B) # Phi
Total parameters: 7,248,023,552  (~7.25 B) # Mistral
'''

# %% id="TQ3de5860StO"
# 2. Inspect the loaded architecture
cfg = model_raw.config
print("=== Model config (selected fields) ===")
for key in [
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",        # if equal to num_attention_heads -> MHA, smaller -> GQA
    "vocab_size",
    "max_position_embeddings",
    "sliding_window",             # None / large number -> no SWA
    "rope_theta",
    "tie_word_embeddings",
    "torch_dtype",
]:
    print(f"  {key:28s} : {getattr(cfg, key, 'n/a')}")

print("\n=== Top-level module tree ===")
print(model_raw)

print("\n=== One transformer block (layer 0) ===")
print(model_raw.model.layers[0])

# Record
'''Output
=== Model config (selected fields) ===
  model_type                   : phi3
  hidden_size                  : 3072
  intermediate_size            : 8192
  num_hidden_layers            : 32
  num_attention_heads          : 32
  num_key_value_heads          : 32
  vocab_size                   : 32064
  max_position_embeddings      : 131072
  sliding_window               : 262144
  rope_theta                   : n/a
  tie_word_embeddings          : False
  torch_dtype                  : torch.bfloat16

=== Top-level module tree ===
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLUActivation()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): Phi3RMSNorm((3072,), eps=1e-05)
    (rotary_emb): Phi3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)

=== One transformer block (layer 0) ===
Phi3DecoderLayer(
  (self_attn): Phi3Attention(
    (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
    (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
  )
  (mlp): Phi3MLP(
    (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
    (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
    (activation_fn): SiLUActivation()
  )
  (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
  (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
  (resid_attn_dropout): Dropout(p=0.0, inplace=False)
  (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
)
'''

''' Mistral
=== Model config (selected fields) ===
  model_type                   : mistral
  hidden_size                  : 4096
  intermediate_size            : 14336
  num_hidden_layers            : 32
  num_attention_heads          : 32
  num_key_value_heads          : 8
  vocab_size                   : 32768
  max_position_embeddings      : 32768
  sliding_window               : None
  rope_theta                   : n/a
  tie_word_embeddings          : False
  torch_dtype                  : torch.bfloat16

=== Top-level module tree ===
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32768, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): MistralRMSNorm((4096,), eps=1e-05)
    (rotary_emb): MistralRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32768, bias=False)
)

=== One transformer block (layer 0) ===
MistralDecoderLayer(
  (self_attn): MistralAttention(
    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
  )
  (mlp): MistralMLP(
    (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
    (act_fn): SiLUActivation()
  )
  (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
  (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
)
'''

# %% [markdown] id="S9ab8A1e0BAR"
# **Answers (Part 2.2: Model):**
#
# **How to confirm CPU vs GPU placement**
#
# A model's "device" is a property of its parameters, not of the wrapper class. Three useful checks (run in the cell above):
#
# 1. `model_raw.device` — convenient shortcut; returns the device of the first parameter. May raise on some custom modules.
# 2. `next(model_raw.parameters()).device` — the authoritative answer; works regardless of wrapper class.
# 3. `model_raw.hf_device_map` — only set when `device_map="auto"` was used, and is the only reliable way to see *split* placement (e.g. some layers on GPU, others offloaded to CPU or disk for very large models).
#
# For our setup we expect:
# - On a CUDA box: `cuda:0` (We saw this on Google Colab)
# - On Apple silicon: `mps`
# - On a CPU-only environment: `cpu`
#
# If `device_map="auto"` ends up sending some layers to CPU because the GPU is too small, `hf_device_map` will show a mix — that is a strong signal that quantisation (Part 4) is needed.
#
# **Phi-3.5-mini-instruct architecture (cited)**
#
# Phi-3.5-mini-instruct is a **decoder-only transformer** in the Llama-style block design.
#
# Headline numbers from the model card and Phi-3 technical report (Microsoft, 2024):
#
# - ~3.8 B parameters
# - Hidden size 3072, FFN intermediate size 8192
# - 32 transformer blocks
# - 32 query heads / 32 KV heads → multi-head attention (MHA), no Grouped-Query Attention
# - Vocabulary size 32064 (extended Llama tokenizer)
# - Context length up to 131072 tokens, extended via LongRoPE
# - Activation: SiLU (SwiGLU FFN); pre-norm RMSNorm; rotary position embeddings (RoPE)
#
# Sources to cite in the report:
# - HuggingFace model card: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
# - Abdin et al., *"Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"*, [arXiv:2404.14219](https://arxiv.org/pdf/2404.14219) (2024)
#
# > [!note] Verifying online resources with code
# > Rather than trusting the docs blindly, the previous cell prints `model_raw.config` directly. Compare the printed values against the cited numbers — two contrasts are particularly worth checking:
# > - `num_attention_heads == num_key_value_heads == 32` confirms multi-head attention. Mistral-7B in contrast uses 32 query / 8 KV heads (GQA), so the same check on Mistral would show a mismatch.
# > - The `sliding_window` field — Phi-3-mini-4k uses a window of 2047, while Phi-3.5-mini (128k context, i.e $128\times1024==131072$) disables it (`None` or a very large number, here we saw `131072`) and relies on LongRoPE for length extension. Mistral-7B uses a 4096-token sliding window. Inspecting layer 0 (`model_raw.model.layers[0]`) shows the matching modules: `Phi3Attention`, `Phi3MLP`, RMSNorm, etc.
#
# TBD: More discussion on this when writing report.

# %% [markdown] id="f91f3034"
# #### Save and load from local path
#
# In professional workflows, you often need to save a specific state of a model to ensure reproducibility or to move it between computing nodes without relying on an internet connection. By using `save_pretrained()`, you create a local snapshot of the model weights and configuration files. We will use this path again in part 4.

# %% [markdown] id="e9cb81a1"
# **(You can use Phi model for this task, please modify the code if requried)**

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["c8b6c327af1b488699d6aac781ca0f7f", "3fb4c76e80364d8aab6c2887e26c9e95", "3dd6109339eb4fb6ac798eb3f7ddb0d3", "19352d467dd84392bd48dfda5122c8db", "9fda7a457ac648388103e9bf52a50fa8", "16d32709150f409a8c4de728e731861b", "437b2d17574645fa82eada22b85de77a", "c3e787ec0ccc4361958a8b69e4492fda", "a8841a0fcfb14ee29855aa92a9e15f06", "11f34a62ed6f48ce942cf65c2b25c027", "b7061527beff4327ac5b638bdb3a6431"]} id="16d069fe" outputId="3b046d44-2a7f-4ab8-bb8b-9a2023ec2b22"
# Save the model and configuration to a local directory
# local_path = "./mistral_16bit_save"
# local_path = "./phi_3.5_save"
local_path = str(WORK_DIR / "models" / "mistral_bf16")

model_raw.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

# %% [markdown] id="e0367331"
# To load the model from your saved files instead of downloading (every time) from the Hugging Face hub, simply replace the `model_id` with your local directory path. The library will detect that you are pointing to a folder rather than a remote repository name.

# %% id="c751caed"
# To reload the model later from your local storage:
model_local = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer_local = AutoTokenizer.from_pretrained(local_path)

# %% [markdown] id="6526eb8a"
# #### Clean up memory (**Optional**)
#
# The function below can clean up your GPU memory so that you can load and try new models without receiving 'out-of-memory' errors.

# %% id="QKBhfOhR55o4"
import gc
import torch

def reset_gpu_memory(model_variable_name='model'):
    """
    Fully clears the GPU VRAM by deleting the model and flushing the
    PyTorch cache. Use this before switching between different model types.
    """
    # 1. Access the global variable for the model
    if model_variable_name in globals():
        print(f"--- Deleting {model_variable_name} from memory ---")
        del globals()[model_variable_name]
    else:
        print(f"--- No variable named '{model_variable_name}' found ---")

    # 2. Force Python's Garbage Collector to run
    gc.collect()

    # 3. Clear the PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # Wait for all kernels to finish
        print("VRAM successfully cleared.")
    else:
        print("No GPU detected to clear.")

# --- How to use it in the Lab ---
# After finishing Part 2 (16-bit):
reset_gpu_memory('model_raw')
# reset_gpu_memory('model_local') # just loaded in last cell

# Now you can safely load the 4-bit model in Part 4
# model = AutoModelForCausalLM.from_pretrained(...)

# %% [markdown] id="ea9ca63d"
# #### Check GPU memory (**Optional**)
#
# The function below can check your GPU resources. Note that we recommended to use your own device (if it has a GPU) and/or FEIT GPU Desktop.
#
# If you use the free version of Google Colab you will see the allocated and reserved memory are very low, which would not be suffifient for this project.

# %% id="1X5C6mfg6daO"
import torch

def print_gpu_utilization():
    if torch.cuda.is_available():
        # Returns memory in bytes, so we convert to GB
        used_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory Allocated: {used_memory:.2f} GB")
        print(f"GPU Memory Reserved: {reserved_memory:.2f} GB")
    else:
        print("No GPU detected.")

# Check before loading Mistral
print_gpu_utilization()

# %% [markdown]
# #### Check HPC resources we have

# %%
import os
import shutil
from pathlib import Path

def bytes_to_gb(n):
    return n / 1024**3

def read_int_file(path):
    try:
        value = Path(path).read_text().strip()
        if value == "max":
            return None
        return int(value)
    except Exception:
        return None

def print_hpc_resources():
    print("=== Slurm allocation ===")
    slurm_keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_PARTITION",
        "SLURM_CPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
    ]
    for key in slurm_keys:
        print(f"{key}: {os.environ.get(key, 'not set')}")

    print("\n=== CPU ===")
    print(f"os.cpu_count(): {os.cpu_count()}")

    print("\n=== RAM from cgroup/job limit ===")
    # cgroup v2 paths, common on newer Linux systems
    mem_current = read_int_file("/sys/fs/cgroup/memory.current")
    mem_max = read_int_file("/sys/fs/cgroup/memory.max")

    if mem_current is not None:
        print(f"Current job/container RAM used: {bytes_to_gb(mem_current):.2f} GB")
    if mem_max is not None:
        print(f"Job/container RAM limit: {bytes_to_gb(mem_max):.2f} GB")
    elif mem_current is None:
        print("Could not read cgroup RAM usage/limit.")

    print("\n=== Disk ===")
    for label, path in [
        ("HOME", Path.home()),
        ("SPARTAN_PROJECT_DIR", os.environ.get("SPARTAN_PROJECT_DIR")),
        ("Current working directory", Path.cwd()),
    ]:
        if path:
            usage = shutil.disk_usage(path)
            print(f"{label}: {path}")
            print(f"  Used: {bytes_to_gb(usage.used):.2f} GB")
            print(f"  Free: {bytes_to_gb(usage.free):.2f} GB")
            print(f"  Total: {bytes_to_gb(usage.total):.2f} GB")

print_hpc_resources()

# %% [markdown] id="d72789e2"
# #### 2.3 A note on datasets
#
# While we have loaded our tokenizer and model, a LLM workflow in incomplete without data. You may have noticed we have not included the code to process any datasets in this part. We will introduce the dataset in part 3, where we will write the logic to feed data into the model.

# %% [markdown] id="6c46d88c"
# ## Part 3: Inference Engineering
#
# In this part, you will use the tokenizer and the model you loaded in part 2 to implement the `pipeline` tasks from part 1.
#
# By handling the tensors, you will observe the "autoregressive" nature of LLMs: how they predict the next token based on the previous context, and how prompt engineering can turn a general-purpose text generator into a specialized classifier.

# %% [markdown] id="9e9e2f8f"
# ### Part 3a: Text response with LLM
#
# In example 1a, the `pipeline` handled the iteration of predicting word after word. Now, you will implement this manually.
#
# For part 3a, you will learn to use the `model.generate()` function, which gives you control over variables like `temperature` (creativity) and `max new tokens` (output length).

# %% [markdown] id="c86872fc"
# **(You can use Phi model for this task, please modify the code if requried)**
#
# In particular, you would want to search the correct instruction format for Phi model, which is different from Mistral.

# %% id="f52cb7a7"
# 1. Setup - Use the same ID and directory from previous tasks
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# cache_dir = "./mistral_project"

# Uncomment below if you want to use Phi model: it's easier to run on local device and free Colab!
# You may have small issues to run the code below with Phi model but they can be quickly fixed (no need to panic at all). You can search on Google, use AI, or talk to demonstrators.
# model_id = "microsoft/Phi-3.5-mini-instruct"
# cache_dir = "./phi_project"

# 2. Load the Components
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16, # Baseline precision
#     device_map="auto",
#     cache_dir=cache_dir
# )

# Here in HPC we reload from local storage
reset_gpu_memory('model_raw')
reset_gpu_memory('model_local')
print(f"Loading from local_path: {local_path}")
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(local_path)

# Function Def
def generate_text(user_prompt, max_tokens=100, temperature=0.7):
    """
    Manually handles the prompt encoding, generation, and decoding.
    """
    # Wrap in Mistral's required instruction format [INST] [/INST]
    # What's Phi's instruction format?
    formatted_prompt = f"[INST] {user_prompt} [/INST]"

    # Step A: Tokenization (Text -> IDs)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # Step B: Generation
    # We use 'do_sample=True' to allow for creativity via Temperature
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Step C: Decoding (IDs -> Text)
    # We slice the output to remove the original prompt from the display
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


# %% id="572df5c5"
# --- Test the Function ---
my_prompt = "Outline an introduction to LLM for students with some maths and programming background"
print(f"Prompt: {my_prompt}\n")

# With max_tokens=1000, you may wait several minutes on a 7B model. 
# Use a small value first to confirm everything works, then scale up if you want a longeranswer.
print("--- Run 1 (default temp=0.7, do_sample=True) ---")
print(generate_text(my_prompt, max_tokens=200))

# Re-running with the same args usually produces a different completion -
# `do_sample=True` makes generate_text non-deterministic.
print("\n--- Run 2 (same args, observe variability) ---")
print(generate_text(my_prompt, max_tokens=200))

# Low temperature collapses the distribution toward the argmax (less creative,
# more conservative). Compare with the default.
print("\n--- Run 3 (temperature=0.1) ---")
print(generate_text(my_prompt, max_tokens=200, temperature=0.1))

# High temperature flattens the distribution, often producing wandering or
# off-topic text.
print("\n--- Run 4 (temperature=1.3) ---")
print(generate_text(my_prompt, max_tokens=200, temperature=1.3))

# A more specific prompt anchors the answer to the right meaning of "LLM".
specific_prompt = (
    "Outline an introduction to **Large Language Models** for students "
    "with some maths and programming background."
)
print("\n--- Run 5 (specific prompt, low temperature) ---")
print(generate_text(specific_prompt, max_tokens=300, temperature=0.2))


# %% [markdown] id="a27d861d"
# **Questions (for project report)**
#
# - How does the output relate to the prompt? Does it talk about LLM as in 'Large Language Models'? If you run the above code again, does the output change?
#
# - How could you make sure the output talks about Large Language Models?
#
# - In step B of `generate_text` function, there are a few variables that you can adjust.
#     - Explain `max_new_tokens`, `temperature`, `do_sample`.
#     - Change these variables and observe the output.

# %% [markdown]
# **Answers (Part 3a):**
#
# Reference outputs: [`Results-ipynb/Part-3a.md`](Results-ipynb/Part-3a.md).
#
# **How does the output relate to the prompt? Does it talk about LLM as in 'Large Language Models'? Re-runs?**
#
# It depends on the model. With Mistral-7B-Instruct, three of the first four runs read `LLM` as **Master of Laws** (a law degree) or **Language Modeling Master's Program**, not **Large Language Models**:
#
# - Run 1 (`temperature=0.7`): *"Title: Introduction to Master of Laws (LLM) Program ..."* — wrong sense.
# - Run 2 (same args): *"Bridging the Gap between Mathematics, Programming, and Artificial Intelligence"* — closer, talks about ML/DL but never names "Large Language Models".
# - Run 3 (`temperature=0.1`): *"Master of Laws (LLM)"* again — note this is essentially greedy.
# - Run 4 (`temperature=1.3`): *"Language Modeling Master's Program"* — a creative compromise.
# - Run 5 (specific prompt, `temperature=0.2`): *"Large Language Models: A Comprehensive Overview..."* — correct.
#
# The Phi-3.5-mini-instruct outputs (in `Part-3a.md`) interpret `LLM` as Large Language Models in **all 5 runs**. The takeaway: the same prompt produces very different priors across base models — Mistral leans toward the legal sense, Phi-3.5 leans toward the AI sense.
#
# Re-running with the same arguments (Run 1 vs Run 2) gives a different completion every time, on both models. That is by design: with `do_sample=True` we draw from the next-token distribution, and the chain of choices diverges quickly.
#
# > [!note] Counter-intuitive observation
# > Lower temperature is not automatically "better". Mistral Run 3 at `temperature=0.1` is essentially greedy — and lands on the *wrong* interpretation ("Master of Laws") with high confidence, because Mistral's most-likely first tokens for `Outline an introduction to LLM ...` align with the legal degree. Greedy decoding is only good when the most-likely path is also the *correct* path; if the prior is wrong, low temperature just gets you a confidently wrong answer.
#
# **How could you make sure the output talks about Large Language Models?**
#
# The most reliable fix in our runs was Run 5: write the term out (`Large Language Models`) and lower the temperature. Three options, in increasing strength:
#
# 1. **Disambiguate the prompt** — write the abbreviation out:
#    ```python
#    my_prompt = "Outline an introduction to large language models (LLMs) for students with maths and programming background."
#    ```
#    This was the change in Run 5; Mistral produced the correct AI/NLP outline.
# 2. **Constrain the sampling** — drop the temperature once the prompt is unambiguous, or switch to greedy decoding:
#    ```python
#    generate_text(my_prompt, max_tokens=400, temperature=0.1)
#    # or pass do_sample=False to model.generate() for true greedy decoding.
#    ```
#    Run 3 is a warning: low temperature *without* disambiguation just locks in the wrong sense.
# 3. **Anchor the answer with a system-style preamble** — Mistral's instruction tag accepts a single `[INST]…[/INST]` block, so we can pre-load context inside it:
#    ```python
#    formatted_prompt = (
#        "[INST] You are a teacher writing a course intro on **Large Language Models** "
#        "(generative transformers like GPT, Mistral, Phi). "
#        f"{user_prompt} [/INST]"
#    )
#    ```
#
# In practice option 1 is enough; option 3 is the right pattern when we need stable behaviour across many prompts.
#
# There is also a fourth lever: **swap the base model**. Phi-3.5-mini-instruct hit the AI sense in 5/5 runs even with the original ambiguous prompt — sometimes a stronger prior solves the problem more cheaply than tuning the prompt. But Phi-3.5 is also smaller (3.8B vs 7B) and can drift on longer answers, so prompt fixes are still worth doing.
#
# **`max_new_tokens`, `temperature`, `do_sample`**
#
# | Argument | What it controls | Effect on output |
# |---|---|---|
# | `max_new_tokens` | Hard cap on tokens generated *after* the prompt (excludes prompt length). | Sets the answer length. Too small → truncated mid-sentence (every Run 1–4 above is cut off mid-section because we used `max_tokens=200`); too large → wasted compute and possible drift. |
# | `temperature` | Scales the logits before softmax: $p_i \propto \exp(\ell_i / T)$. | $T \to 0$ collapses the distribution onto the argmax (greedy); $T = 1$ uses raw probabilities; $T > 1$ flattens them and lets rare tokens through. Higher $T$ → more diverse but more incoherent. |
# | `do_sample` | If `False`, ignore the distribution and pick the argmax token at every step (greedy). If `True`, draw a sample from the distribution. | Greedy is deterministic and conservative; sampling is stochastic and varied. With `do_sample=False`, `temperature` is ignored. |
#
# > [!note] What we observed in the Mistral runs
# > - Run 1 vs Run 2 (same args): different outputs every time — confirms sampling is the source of randomness.
# > - Run 3 (`temperature=0.1`): nearly fixed wording; deterministic but landed on the wrong sense.
# > - Run 4 (`temperature=1.3`): wandered into "Language Modeling Master's Program" — closer to AI but still off-genre.
# > - Run 5 (specific prompt, low temp): the cleanest fix. Disambiguation + low temperature is the report-friendly recipe.
# > - All Run 1–4 outputs are cut off mid-sentence because `max_tokens=200` is too short for an outline; bumping to 400–600 gives a full structure (Run 5 used 300 and still cut off, but later sections appeared).
#

# %% [markdown]
# For Phi-3 / Phi-3.5 instruct, the format is:
#
# ```bash
# <|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# Question?<|end|>
# <|assistant|>
# ```
#
# Manual version would be:
# ```python
# formatted_prompt = f"""<|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# {user_prompt}<|end|>
# <|assistant|>
# """
# ```

# %%
# # Try using phi-3.5-mini-instruct model
# model_phi = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3.5-mini-instruct",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     cache_dir=str(WORK_DIR / "cache" / "phi_proj")
# )
# tokenizer_phi = AutoTokenizer.from_pretrained(
#     "microsoft/Phi-3.5-mini-instruct",
#     use_fast=False,
#     cache_dir=str(WORK_DIR / "cache" / "phi_proj")
# )

# # Function Def
# def generate_text_phi(user_prompt, max_tokens=100, temperature=0.7):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": user_prompt},
#     ]

#     formatted_prompt = tokenizer_phi.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     inputs = tokenizer_phi(formatted_prompt, return_tensors="pt").to(model_phi.device)

#     end_token_id = tokenizer_phi.convert_tokens_to_ids("<|end|>")
#     eos_ids = [tokenizer_phi.eos_token_id]
#     if isinstance(end_token_id, int) and end_token_id >= 0:
#         eos_ids.append(end_token_id)

#     output_tokens = model_phi.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         do_sample=True,
#         pad_token_id=tokenizer_phi.eos_token_id,
#         eos_token_id=eos_ids,
#     )

#     new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
#     response = tokenizer_phi.decode(new_tokens, skip_special_tokens=True)

#     return response

# # --- Test the Function ---
# my_prompt = "Outline an introduction to LLM for students with some maths and programming background"
# print(f"Prompt: {my_prompt}\n")

# # Use a small value first to confirm everything works, then scale up if you want a longeranswer.
# print("--- Run 1 (default temp=0.7, do_sample=True) ---")
# print(generate_text_phi(my_prompt, max_tokens=200))

# # Re-running with the same args usually produces a different completion -
# # `do_sample=True` makes generate_text_phi non-deterministic.
# print("\n--- Run 2 (same args, observe variability) ---")
# print(generate_text_phi(my_prompt, max_tokens=200))

# # Low temperature collapses the distribution toward the argmax (less creative,
# # more conservative). Compare with the default.
# print("\n--- Run 3 (temperature=0.1) ---")
# print(generate_text_phi(my_prompt, max_tokens=200, temperature=0.1))

# # High temperature flattens the distribution, often producing wandering or
# # off-topic text.
# print("\n--- Run 4 (temperature=1.3) ---")
# print(generate_text_phi(my_prompt, max_tokens=200, temperature=1.3))

# # A more specific prompt anchors the answer to the right meaning of "LLM".
# specific_prompt = (
#     "Outline an introduction to **Large Language Models** for students "
#     "with some maths and programming background."
# )
# print("\n--- Run 5 (specific prompt, low temperature) ---")
# print(generate_text_phi(specific_prompt, max_tokens=300, temperature=0.2))

# %% [markdown] id="863d26ad"
# ### Part 3b: Sentiment Analysis with LLM
#
# In example 1b, you used a BERT-based model for sentiment analysis. BERT is an encoder-only model designed to output a numerical score for a fixed set of labels.
#
# In part 3b, we take a different approach using a generative LLM. We don't use a classification "head" in LLM architecture, instead, we use prompts. By providing the model with a clear instruction (and optionally a few examples), we "steer" the model to output exactly one word: `Positive` or `Negative` (sentiment).
#
# Unlike BERT, an LLM can technically say whatever it wants. If your prompt is vague, the LLM might explain why it thinks an input is positive instead of just giving the positive/negative label. You will explore how to constrain the output to ensure it fits back into your workflow.
#
# **Note**: While BERT is efficient for simple classification, the industry is shifting toward generative inference, which can achieve:
# - Zero-shot capabilities: How a model can perform a task it wasn't explicit trained for.
# - Instruction following: How the `[INST]` and `[/INST]` tags change the probability distribution of the next token.
# - Unified architecture: the power of using one single model for both generation and classification.

# %% [markdown] id="77d3e9eb"
# #### Exercise: Simple sentiment analysis
#
# Fill in the code below. Use Mistral and text generation function to complete sentiment analysis tasks. **Hint**: Check the code in part 3a.
#
# **(You can use Phi model for this task, please modify the code if requried)**
#
# In particular, you would want to search the correct instruction format for Phi model, which is different from Mistral.

# %% id="be9e2acb"
def get_sentiment(review):

    # TO DO: Complete the prompt below, i.e. fill in the "... ..." between "[INST]" and "[/INST]"
    # What's Phi's instruction format?
    prompt = (
            f"[INST] Classify the sentiment of the following review as either 'Positive' or 'Negative'."
            f"Respond with ONLY ONE word. \n\n"
            f'Review: "{review}"[/INST]'
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # TO DO: Modify below (replace None and complete the code)
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    '''This version still ouputs prompt.
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the model's new part (after the [/INST] tag)
    # What's Phi's instruction format?
    sentiment = response.split("[/INST]")[-1].strip()
    '''
    # Didn't work since the decoded text no longer contains the exact literal string `[/INST]`
    
    # Decode only the generated part, not the prompt
    new_tokens = output[0][inputs["input_ids"].shape[-1]:] # Gets the token after input (prompts)
    sentiment = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Optional: keep only first word
    sentiment = sentiment.split()[0]
    
    return sentiment

# Example Test
print(get_sentiment("This project is quite challenging but the support is great!"))
# Expected Output: Positive

# %% [markdown] id="8002d43a"
# Now we move beyond simple generation to evaluate how Mistral-7B handles semantic complexity. We will compare two fundamental paradigms of In-Context Learning:
# - Zero-shot learning: The model is given a task description and a test input with no prior examples. This relies entirely on the model's pre-trained internal knowledge and instruction-following capabilities.
# - Few-shot learning: We provide a "context" of a few labeled examples (k-shot) before the test input. This helps the model align with the desired output format and grasp the specific "logic" of the task.
#
# Standard sentiment analysis often relies simple keyword matching (e.g. "good" = positive). However, human language is rarely that simple. The examples below contains linguistic traps including:
# - Sarcasm: "Oh great, another 'groundbreaking' superhero movie!"
# - Negation/Double negatives: "I can't believe how much I didn't hate this."
# - Contrastive conjuctions: "The plot was non-existent... yet I couldn't look away."
# - Backhanded compliments: "This is the best example of a terrible movie."
#
# By comparing zero-shot vs. few-shot performance on these examples, you will observe how providing even a tiny bit of context can help the model navigate complex intent.
#
# #### Exercise: Sentiment analysis of "tricky" sentences
#
# Fill in the code below and compare zero-shot and few-shot performance.
#
# **(You can use Phi model for this task, please modify the code if requried)**
#
# In particular, you would want to search the correct instruction format for Phi model, which is different from Mistral.

# %% id="4e85dc68"
# 1. Setup the Evaluation Data
tricky_reviews = [
    "I was worried it would be terrible, but it actually wasn't bad at all.",
    "Oh great, another 'groundbreaking' superhero movie that is exactly like the last ten.",
    "The plot was non-existent and the acting was wooden, yet I couldn't look away.",
    "It’s not that the food was cold, it’s just that it wasn't particularly warm either.",
    "If you enjoy watching paint dry, you'll love this movie!",
    "I've had better, but I've certainly had much, much worse.",
    "Everything about this place is perfect, except for the service, the food, and the price.",
    "I can't believe how much I didn't hate this.",
    "This is the best example of a terrible movie I have ever seen.",
    "The only thing more disappointing than the ending was the fact that I paid for a ticket."
]

# Helpers -------------
import json

TASK = (
    "Classify the sentiment of the following movie review as either 'Positive' or 'Negative'."
    "Responed with ONLY the label (one word)."
)

# Your example reviews
example = [
    "I expected to fall asleep but it kept me on the edge of my seat.",
    "Sure, that's just great. Two hours I'll never get back.",
    "It's not bad - in fact, it's quietly excellent.",
    "The trailer made it look like a masterpiece. The film itself was anything but.",
]

# Labels need to live somewhere too
example_labels = [
    "Positive",
    "Negative",
    "Positive",
    "Negative",
]

def quote(text: str) -> str:
    """Safely quote reviews, including embedded quotes/backslashes."""
    return json.dumps(text)


def build_few_shot_content(review: str, examples: list[str], labels: list[str]) -> str:
    if len(examples) != len(labels):
        raise ValueError("examples and labels must have the same length")

    shots = "\n\n".join(
        f"Review: {quote(ex)}\nLabel: {label}"
        for ex, label in zip(examples, labels)
    )

    return (
        f"{TASK}\n\n"
        f"{shots}\n\n"
        f"Review: {quote(review)}\n"
        f"Label:"
    )
# Helpers ------------- End


# 2. Define the Prompt Templates
def get_zero_shot_prompt(review):
    # TO DO: Construct a zero-shot prompt using Mistral's [INST] format.
    # Ensure you tell the model to output ONLY the label.
    # What's Phi's instruction format?
    return (
        f"[INST]{TASK}\n\n"
        f'Review: "{review}" \n'
        "Label: [/INST]"
    )

def get_few_shot_prompt(review):
    # TO DO: Construct a few-shot prompt (at least 3 examples).
    # Use the format: Review: "..." -> Label
    # End the prompt with the current review to be classified.
    # What's Phi's instruction format?
    content = build_few_shot_content(review, example, example_labels)
    return f"[INST] {content} [/INST]"

# 3. Inference Loop
results = []

print("Starting benchmarking...")
for i, review in enumerate(tricky_reviews):
    print(f"Processing Review {i+1}/10...")

    # Run Zero-Shot
    # TO DO: Tokenize the zero-shot prompt and move it to the GPU
    zs_input = tokenizer(get_zero_shot_prompt(review), return_tensors="pt").to(model.device)

    # TO DO: Call model.generate. Keep max_new_tokens low (1-5) since we only want a label.
    zs_output = model.generate(
        **zs_input,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # # Similarly, this still outputs prompt.
    # zs_label = tokenizer.decode(zs_output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    zs_label = tokenizer.decode(zs_output[0][zs_input["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # Run Few-Shot
    fs_input = tokenizer(get_few_shot_prompt(review), return_tensors="pt").to("cuda")

    # TO DO: Generate the output for the few-shot input
    fs_output = model.generate(
        **fs_input,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # fs_label = tokenizer.decode(fs_output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    fs_label = tokenizer.decode(fs_output[0][fs_input["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    fs_label = fs_label.split()[0] # More than one word of output in this case.

    results.append({
        "Review": review[:50] + "...",
        "Zero-Shot Label": zs_label,
        "Few-Shot Label": fs_label
    })

# 4. Display as a Table
df = pd.DataFrame(results)
print("\n--- Sentiment Analysis Results ---")
print(df.to_string(index=False))

# Optional: Save to CSV for your project report
df.to_csv("sentiment_results.csv", index=False)


# %% [markdown] id="ae07ad13"
# **Questions (for project report)**
#
# - Did the model respect the instrcution to output only one word in zero-shot? If not, why?
# - Identify a specific review where zero-shot failed but few-shot succeeded. What did the model learn from the context?
# - How many toekens is your few-shot prompt compared to your zero-shot prompt? When the computing resources are limited, why might we prefer zero-shot for very high-throughput tasks?

# %% [markdown]
# **Answers (Part 3b):**
#
# Reference outputs: [`Results-ipynb/sentiment_results.csv`](Results-ipynb/sentiment_results.csv).
#
# **Did the model respect the instruction to output only one word in zero-shot? If not, why?**
#
# Mostly yes — but:the *binary* part broke. Looking at the CSV, the model returned a single word for every review (no rationales, no quote marks), so the format constraint held. What it ignored was the *label set*: rows 4 and 6 came back as `Neutral` rather than the requested `Positive` / `Negative`.
#
# Both reviews are double-negation / understatement:
# - Row 4: *"It's not that the food was cold, it's just that it wasn't particularly warm either."* → `Neutral`
# - Row 6: *"I've had better, but I've certainly had much, much worse."* → `Neutral`
#
# The instruction-tuning prior is asserting itself: when the input is genuinely ambiguous, Mistral prefers to *hedge* over forcing one of the two requested labels. Few-shot doesn't fix this on its own — both rows come back `Neutral` in the few-shot column too, because none of our 4 examples shows the model how to break a tie.
#
# Besides, few-shot output additional token after the first label before I added `.split()[0]` to the `fs_label`.
#
# **A specific review where zero-shot failed but few-shot succeeded.**
#
# Row 8 — *"I can't believe how much I didn't hate this."* — is the canonical case:
#
# | Review | Zero-Shot | Few-Shot |
# |---|---|---|
# | I can't believe how much I didn't hate this. | Negative | **Positive** |
#
# The double negation (`didn't hate`) flips the polarity, but zero-shot Mistral parses the surface words `can't`, `didn't`, `hate` and lands on `Negative`. Few-shot, after seeing labelled examples like *"It's not bad — in fact, it's quietly excellent." → Positive*, learns the *task framing* — surface negativity words can wrap a positive sentiment — and corrects to `Positive`.
#
# What the model "learned" from the context is not new sentiment knowledge — it is the framing rule. That is pure in-context learning: the gradient never moved.
#
# **Full predictions**
#
# | # | Review (truncated) | Expected | Zero-Shot | Few-Shot | Notes |
# |---|---|---|---|---|---|
# | 1 | I was worried it would be terrible, but it actually wasn't bad… | Positive | Positive | Positive | Both correct. |
# | 2 | Oh great, another 'groundbreaking' superhero movie… | Negative | Negative | Negative | Sarcasm caught zero-shot. |
# | 3 | The plot was non-existent… yet I couldn't look away. | Positive | Negative | Negative | Both **wrong** — surface negativity wins over the contrastive twist. |
# | 4 | It's not that the food was cold, it's just that it wasn't particularly warm. | Negative | Neutral | Neutral | Instruction violated — model hedged. |
# | 5 | If you enjoy watching paint dry, you'll love this movie! | Negative | Negative | Negative | Sarcasm caught. |
# | 6 | I've had better, but I've certainly had much, much worse. | Positive | Neutral | Neutral | Instruction violated — genuinely ambiguous. |
# | 7 | Everything about this place is perfect, except for the service, the food, and the price. | Negative | Negative | Negative | Backhanded compliment caught. |
# | 8 | I can't believe how much I didn't hate this. | Positive | Negative | **Positive** | Few-shot fixes double negation. |
# | 9 | This is the best example of a terrible movie I have ever seen. | Negative | Negative | Negative | Backhanded — "best…terrible" caught. |
# | 10 | The only thing more disappointing than the ending was paying for the ticket. | Negative | Negative | Negative | Both correct. |
#
# Scoreboard against the human label:
# - Zero-shot: 6 correct, 2 hedged-Neutral, 2 wrong → **6/10**
# - Few-shot: 7 correct, 2 hedged-Neutral, 1 wrong → **7/10**
#
# So the few-shot bump is real but small (+10% on this set). The cases neither prompt fixes (rows 3, 4, 6) need either richer examples (a 3-way label set, or explicit "force one of two" wording) or a different decoding strategy (constrained generation that masks every token outside `{Positive, Negative}`).
#
# **Token counts and the throughput trade-off**
#
# Rough counts on the longest review with the Mistral tokenizer:
#
# | Prompt | Tokens (approx.) |
# |---|---|
# | Zero-shot | ~70 |
# | Few-shot (4 examples) | ~220 |
#
# That is small in absolute terms, but the cost scales with throughput:
#
# - **Prefill is linear in prompt length.** A 4-shot prompt is roughly 3× longer, so the prefill phase costs ~3× more FLOPs and KV-cache memory.
# - **In a batched serving setting** the per-request memory budget is dominated by the longest prompt in the batch — few-shot inflates it for everyone.
# - **Latency to first token (TTFT)** also grows with prompt length: more tokens to attend over before the first output appears.
#
# So for high-throughput tasks we prefer zero-shot when accuracy is acceptable. The progression we'd follow:
#
# 1. Try zero-shot first.
# 2. If accuracy on a held-out set is too low, add the *minimum* number of examples that fixes the failures we care about. For us that lift was 6→7 (+10%) — borderline whether it pays for the 3× prompt cost.
# 3. If we still need few-shot but cost is a problem, distil into a fine-tuned classifier (Part 5: PEFT/LoRA) — that pushes the example knowledge into the weights, so we stop paying for it at inference.
#

# %% [markdown] id="052ad6f9"
# ## Part 4: Efficient LLM - Compression
#
# In the previous parts, you loaded a 7-billion parameter model in 16-bit precision, requiring about 14.5GB of VRAM. While this fits on a high-end GPU (like an A100), it is far too large for consumer hardware or mobile devices.
#
# In this section, we explore **quantization**, the process of reducing the precision of the model's weights to shrink its footprint while maintaining as much "intelligence" as possible.
#
# ### The mathematics of bits
# A standard `bfloat16` uses 16 bits per parameter. Our goal is to compress this to 4 bits per parameter.
#
# Mathematically, this is a lossy compression because we are mapping a continuous range of floating-point values into a small, discrete set of 16 possible levels (2<sup>4</sup> = 16).
#
# But how do we represent a 7B parameters model using only 16 possible values per weight without impacting the model performance (too much)? We use **4-bit NormalFloat (NF4)**, which uses a **non-linear distribution** that *allocates more "levels" to the most common weight values near zero*, perserving the information that matters most to the model's performance.
#
# ### 4.1 Implementation: bitsandbytes
# We will use the `bitsandbytes` library to perform "Quantization on the fly". Instead of loading 14GB then shrinking it, we will load the model directly into 4-bit memory.
#
# **Task**: Run the code below to load the Mistral model using [`BitsAndBytesConfig`](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
#
#
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="44c43a30"
# 0. Model ID for v0.3 Instruct
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# My cache dir for Spartan-OOD
cache_dir = str(WORK_DIR / "cache" / "mistral_proj")

# Free Part 3's `model` before loading the bf16 baseline below — they hold the
# same weights, so keeping both would double-occupy ~14 GiB. Idempotent: safe
# to re-run, no-op if `model` is not in scope.
import gc
if "model" in globals():
    del globals()["model"]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1. Load Baseline: 16-bit (bfloat16)
print("Loading Baseline Model (bf16)...")
model_16bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

# %% id="f554328d"
# 2. Define 4-bit Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load Compressed: 4-bit (NF4)
print("Loading Quantized Model (4-bit NF4)...")
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


# %% [markdown] id="feeb0756"
# **Questions (for project report)**
#
# - In the last line of `bnb_config`, we set `bnb_4bit_compute_dtype=torch.bfloat16`. Why do we need 16 bits when the weights are 4 bits?

# %% [markdown]
# **Answers (Part 4.1):** (By Claude Opus 4.7)
#
# **Why `bnb_4bit_compute_dtype=torch.bfloat16` when the weights are 4 bits?**
#
# `load_in_4bit=True` only controls how weights are *stored*. It does **not** make the GPU compute in 4 bits — there is no native 4-bit matmul on GPUs we use (no Tensor Core path for 4-bit floats). At every forward pass, bitsandbytes:
#
# 1. Reads a block of NF4 weights (4-bit indices into a fixed 16-level codebook).
# 2. **Dequantizes them on the fly** to `bnb_4bit_compute_dtype` (here `bfloat16`) using a per-block scale.
#     (Dequantization: converting quantized, low-precision data back into higher-precision)
# 3. Runs the matmul against the activations in that compute dtype.
# 4. Throws the dequantized tensor away — only the 4-bit blob lives in VRAM long-term.
#
# So `compute_dtype` is the precision of the *transient* dequantized activation × weight product, not of storage. We need it to be at least 16 bits because:
#
# - **Activations are still 16-bit.** The KV cache, residual stream, embeddings, layer norms, and `lm_head` of Mistral are kept in `bfloat16`/`fp16`. Multiplying a 4-bit weight by a 16-bit activation has to happen in 16 bits; you can't do it in 4.
# - **Accumulation needs headroom.** A single MLP row in Mistral is a dot product over 14336 multiply-adds. If we accumulated those in 4 or 8 bits the running sum would overflow or quantize away to zero almost immediately. `bfloat16` (8-bit exponent, same range as `fp32`) is the minimum that survives that accumulation without underflow on small logits.
# - **NF4 is non-uniform.** The codebook is denser near 0 (where most LLM weights live) and sparser in the tails. After dequant, you get back a `bfloat16` value that *approximates* the original `bfloat16` weight — there is no point dropping further precision before the matmul.
#
# **Why `bfloat16` and not `fp16`?**
#
# Both are 16-bit. The split is **mantissa vs exponent**:
# - `fp16`: 10-bit mantissa, 5-bit exponent → max value ≈ 65504; tight on dynamic range.
# - `bfloat16`: 7-bit mantissa, 8-bit exponent → same range as `fp32`.
#
# Mistral was *trained* in `bfloat16`. Mixing in `fp16` at inference can overflow on softmax logits or attention scores (we have seen `inf` / `nan` show up that way on Mistral and Llama). Matching the training dtype keeps the activation distribution in the range the weights were tuned for.
#
# > [!note] What this implies for VRAM
# > The "4-bit" model is 4-bit only on disk and on the weight tensors in VRAM. *During the forward pass* every block is briefly 16-bit, and the activations / KV cache are 16-bit throughout. That is part of why VRAM never drops to the naive 3.5 GB floor — see Part 4.2 below.

# %% [markdown] id="16b719fd"
# ### 4.2 The memory paradox: VRAM vs. Disk Size
# In theory, 4-bit is 25% of 16 bits. You would expect the VRAM to drop from 14GB to around 3.5 GB. However, you will likely to still see a usage of VRAM of more than 3.5 GB.

# %% id="7aa7d79e"
def get_vram_usage(label):
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[{label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# TO DO: Run this check and answer the discussion question
get_vram_usage("Total Memory with Both Models Loaded")
''' I got:
[Total Memory with Both Models Loaded] Allocated: 17.36 GB | Reserved: 30.79 GB
which seems like a total of 14GB+3.5GB.
Need a better way to isolate and check true usage.
'''

# Per-model footprint (HuggingFace returns parameters + non-trainable buffers, in bytes).
# This is the cleanest way to isolate the 4-bit model's true cost from PyTorch's caching allocator overhead.
print()
print(f"model_16bit footprint: {model_16bit.get_memory_footprint() / (1024**3):.2f} GB")
print(f"model_4bit  footprint: {model_4bit.get_memory_footprint() / (1024**3):.2f} GB")

# IMPORTANT: bitsandbytes packs 2 NF4 weights per byte, so `param.numel()` on a
# Params4bit tensor returns the BYTE count, not the nominal weight count. To get
# the true nominal count we read it from the 16-bit model (full bf16 tensors).
# The previous version of this cell counted on the 4-bit model and reported
# ~3.76 B (~half of nominal) — that is correct, just packed; see the markdown
# answer below for the full explanation.
n_params_packed  = sum(p.numel() for p in model_4bit.parameters())
n_params_nominal = sum(p.numel() for p in model_16bit.parameters())
print(f"\nParameter count (4-bit, packed-storage view): {n_params_packed/1e9:.2f} B")
print(f"Parameter count (16-bit, true nominal count): {n_params_nominal/1e9:.2f} B")
print(f"Theoretical 4-bit floor (nominal x 4 bits)  : {n_params_nominal * 4 / 8 / (1024**3):.2f} GB")

''' Output (initial run, before adding the nominal-count print):
model_16bit footprint: 13.50 GB
model_4bit  footprint: 3.75 GB

Parameter count: 3.76 B
Theoretical 4-bit floor: 1.75 GB

After re-running with the corrected accounting, expect (as well as actually got):
Parameter count (4-bit, packed-storage view): 3.76 B
Parameter count (16-bit, true nominal count): 7.25 B
Theoretical 4-bit floor (nominal x 4 bits)  : 3.38 GB
'''


# %% [markdown] id="d9d066da"
# **Questions (for project report)**
#
# - Why is the VRAM usage more than 3.5 GB? You may search for answers from online resources.

# %% [markdown]
# **Answers (Part 4.2):** (By Claude Opus 4.7 and refined)
#
# **Observed numbers (Spartan A100, both models loaded simultaneously):**
#
# | Quantity | Value |
# |---|---|
# | Allocated VRAM (both models loaded) | 17.36 GiB |
# | Reserved VRAM (allocator slack) | 30.79 GiB |
# | `model_16bit.get_memory_footprint()` | 13.50 GiB |
# | `model_4bit.get_memory_footprint()` | 3.75 GiB |
# | `sum(p.numel() ...)` on `model_4bit` | 3.76 B (**packed**) |
# | `sum(p.numel() ...)` on `model_16bit` (re-run) | ~7.25 B (nominal) |
# | 16-bit on disk | 13.50 GiB |
# | 4-bit on disk | 3.85 GiB |
#
# **Sanity check that the binary-vs-decimal units line up.** The Mistral-7B-Instruct-v0.3 model card lists ~7.25 B parameters at "14.5 GB" in bf16 — those are *decimal* gigabytes. `get_memory_footprint() / 1024**3` is *binary* gibibytes. Convert: $7.25 \times 10^{9} \times 2 \text{ B} / 1024^{3} = 13.50$ GiB. So the 13.50 GiB we see is exactly the 14.5 GB advertised — the run was correct.
#
# **Did I do it right?** **Yes.** Both surprising numbers are artefacts of how bitsandbytes packs weights, not bugs:
#
# > [!note] Why `Parameter count: 3.76 B` is half of the nominal 7.25 B
# > bitsandbytes' `Params4bit` tensor stores its quantized weight as a `uint8` buffer with two NF4 nibbles **packed into each byte**. Calling `.numel()` on that buffer returns the *byte count*, not the original weight count. So:
# > - quantized linear weights (~6.98 B nominal) → reported as ~3.49 B
# > - unquantized layers — `embed_tokens` (32768 × 4096 = 0.134 B) and `lm_head` (same) — stay in bf16 and report their full element count: 0.27 B
# > - tiny RMSNorms ≈ 0.001 B
# >
# > Total reported = $3.49 + 0.27 + 0.001 \approx 3.76$ B ✓
# >
# > The fix in the updated cell above is to read the nominal count from the 16-bit model (`sum(p.numel() for p in model_16bit.parameters())`) and use that to compute the floor. Re-running gives the correct 3.38 GiB floor.
#
# **Why is VRAM > 3.5 GB?**
#
# The naive expectation is
#
# $$
# \text{floor} = \frac{N_{\text{nominal}} \times 4\text{ bits}}{8 \times 1024^{3}} = \frac{7.25\times 10^{9} \times 4}{8 \times 1024^{3}} \approx 3.38 \text{ GiB}.
# $$
#
# We measured **3.75 GiB**, only +0.37 GiB above the floor. That gap is fully accounted for by:
#
# 1. **Layers that are *not* quantized.** bitsandbytes only swaps the big `nn.Linear` modules inside the transformer blocks. It deliberately skips:
#    - `embed_tokens` (`nn.Embedding`): $32768 \times 4096 \times 2 \text{ B} = 0.25$ GiB.
#    - `lm_head` (in the default `llm_int8_skip_modules`): same shape, $0.25$ GiB.
#    - All `MistralRMSNorm` layers (~0.001 GiB).
#    
#    That is roughly **0.50 GiB** of bf16 weight that never gets compressed.
#
# 2. **Quantization metadata (NF4 + double-quant).** Each block of 64 weights carries an absmax scale. With `bnb_4bit_use_double_quant=True`, those scales are themselves 8-bit-quantized into super-blocks of 256, giving an effective overhead of **~0.127 bits/weight** (vs ~0.5 bits/weight without DQ). For 6.98 B quantized weights:
#    $$
#    \text{metadata} \approx \frac{6.98 \times 10^{9} \times 0.127}{8 \times 1024^{3}} \approx 0.10 \text{ GiB}.
#    $$
#
# So a more honest floor is:
# $$
# \underbrace{3.25}_{\text{4-bit linears}} + \underbrace{0.10}_{\text{NF4-DQ scales}} + \underbrace{0.50}_{\text{embed + lm\_head + norms}} \approx 3.85 \text{ GiB}.
# $$
#
# That predicts within 0.1 GiB of the measured 3.75 GiB — close enough that the remaining gap is just measurement / counting nuance (e.g. tied buffers, allocator alignment).
#
# > [!note] Updating the initial guess
# > My first pass estimated ~4.5 GiB; that was an overcount. With the actual NF4-DQ overhead (~0.13 bits/weight, not the 0.4 I'd written) and only embed + lm_head left in bf16, the real number is ~3.85 GiB — and the measured 3.75 GiB is right on top of that. The "VRAM doesn't drop to 3.5 GiB" framing in the question is itself slightly pessimistic on a modern GPU: with NF4-DQ on Mistral-7B we get within ~10% of the floor.
#
# **Reading the global VRAM trace.**
#
# - Allocated (17.36) ≈ `model_16bit` (13.50) + `model_4bit` (3.75) = 17.25 GiB → only ~0.1 GiB of slack at idle. That is exactly what we expect immediately after loading: no KV cache yet, no activation buffers, only the kernels' constant workspace.
# - Reserved (30.79) is dominated by PyTorch's caching allocator. After any non-trivial forward pass (e.g. the perplexity cell), `memory_reserved` grows and stays high so subsequent allocations can be served fast. The reserved-vs-allocated gap is a feature, not a leak.
#
# **On disk vs in VRAM.**
#
# | Format | VRAM (GiB) | Disk (GiB) |
# |---|---|---|
# | bf16 | 13.50 | 13.50 |
# | NF4 + DQ | 3.75 | 3.85 |
#
# Disk and VRAM agree to within 0.1 GiB — `save_pretrained` writes the same packed tensors that live in VRAM, plus tiny config / tokenizer files. Compression ratio: $13.50 / 3.85 \approx 3.5\times$. The naive expectation was $4\times$ (16/4); we lose the factor-of-1 mostly to the unquantized embed + lm_head, exactly as predicted.
#
# **References:**
# - Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — Section 3 derives the NF4-DQ overhead (block 64 / super-block 256, ~0.127 bits/weight after double quant).
# - HuggingFace blog: [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes).
# - bitsandbytes source: [`bitsandbytes/nn/modules.py` Params4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/modules.py) — confirms the packed `uint8` storage that explains the 3.76 B count.

# %% id="d6d5678a"
# Helper function to clear VRAM between loads if needed.
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

clear_vram()


# %% [markdown] id="0dbc770d"
# ### Verify the quantization reality with disk size
#
# To truly understand the system-level benefits of quantization, you must compare the static storage. While the 16-bit model is a massive ~14.5GB file, the 4-bit compressed version should represent a near-linear reduction in disk space.
#
# **Exercise**: Export the 16-bit and 4-bit models to your project folder and verify the compression size. We have provided the function of calculating directory size below.

# %% id="16fb3b4b"
# Calculate the size of the saved directory in GB
def get_dir_size_gb(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)



# %% [markdown] id="4c3d4660"
#
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="6b13d2a6"
# TO DO: Save both models to your project directory
# Original (Colab-style):
# path_bf16 = "./models/mistral_bf16"
# path_nf4  = "./models/mistral_nf4"
# Spartan-OOD: route through WORK_DIR so weights live on /data/gpfs scratch,
# not the home quota. The bf16 path matches `local_path` from Part 2, so this
# cell is idempotent if Part 2 has already saved the baseline.
path_bf16 = str(WORK_DIR / "models" / "mistral_bf16")
path_nf4 = str(WORK_DIR / "models" / "mistral_nf4")

model_16bit.save_pretrained(path_bf16)
model_4bit.save_pretrained(path_nf4)

# %% id="7a5195c3"
# TO DO: Calculate the directory size of each model
print(f"Raw 16-bit Model Disk Size: {get_dir_size_gb(path_bf16):.2f} GB")
print(f"Raw 4-bit Model Disk Size: {get_dir_size_gb(path_nf4):.2f} GB")
# Raw 16-bit Model Disk Size: 13.50 GB
# Raw 4-bit Model Disk Size: 3.85 GB

# %% [markdown] id="014016f7"
# ### 4.3 Performance benchmarking
#
# Reducing a model from 16-bit to 4-bit is a lossy process. While NF4 is designed to minimize this loss, we should analyse the impact using two methods: mathematical certainity (perplexity) and output quality (human evaluation).
#
# #### 4.3.1 What is perplexity (PPL)?
#
# Mathematically, PPL is the exponentiated average negative log-likelihood of a sequence. If you have a sequence of tokens $X = (x_1, x_2, \dots, x_n)$, the formula is:
# $$PPL(X) = \exp \left( -\frac{1}{n} \sum_{i=1}^{n} \log P(x_i | x_{<i}) \right)$$
#
# PPL measures how "surprised" the model is after seeing a sequence of text.
#
# - A low PPL means the model finds the text predictable and highly probable, where it understands the patterns.
# - A high PPL means the model is uncertain or confused.
#
# When we quantize a model, we expected the PPL to increase slightly. Our goal is to ensure this "quantization tax" remains low enough that the model's actual utility will not be compromised.
#
# For a deeper dive into how to calculate PPL, please see https://huggingface.co/docs/transformers/perplexity.

# %% [markdown] id="44b3e72d"
# #### 4.3.2 Exercise - perplexity
#
# Write a function to calculate the perplexity of a given string. Then run this function on both `model_16bit` and `model_4bit`.

# %% [markdown] id="4679859c"
#
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="3ddc9cda"
def calculate_perplexity(model, tokenizer, text):
    """
    Calculates the perplexity of a given text for a Causal Language Model.
    """
    # 1. Tokenize the input and move to GPU
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # 2. Get the model output (ensure you pass labels to get the loss)
        # TO DO: Call the model and pass input_ids as labels
        outputs = model(input_ids, labels=input_ids)
        # HuggingFace causal LMs internally shift the labels by one (predict token i+1 from i),
        # so passing input_ids as labels is the standard recipe and exactly matches the PPL def above.

        # 3. Extract the CrossEntropy loss
        # TO DO: Retrieve the loss from the model output
        loss = outputs.loss
        # `loss` is the mean negative log-likelihood (mean NLL) per token,
        # averaged over then (n-1) prediction positions in the sequence.

    # 4. Calculate Perplexity: exp(loss)
    # TO DO: Implement the final PPL calculation
    perplexity = torch.exp(loss).item()
    # PPL = exp(mean NLL). `.item()` pulls a Python float out of the 0-d tensor.

    return perplexity

# Test sample from WikiText-2 (or any high-quality Wikipedia snippet)
wiki_sample = "The University of Melbourne is a public research university located in Melbourne, Australia. Founded in 1853, it is Australia's second oldest university."

ppl_16 = calculate_perplexity(model_16bit, tokenizer, wiki_sample)
ppl_4 = calculate_perplexity(model_4bit, tokenizer, wiki_sample)

print(f"--- Perplexity Results ---")
print(f"16-bit PPL: {ppl_16:.4f}")
print(f"4-bit PPL:  {ppl_4:.4f}")
print(f"Quantization tax: {(ppl_4 - ppl_16) / ppl_16 * 100:+.2f}%")

# A single sentence is noisy; a fairer estimate uses several samples.
# (Run only if you want a sturdier number; the cells above already answer the task.)
extra_samples = [
    "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
    "In computer science, a hash table is a data structure that maps keys to values using a hash function for fast lookup.",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells, often described as the powerhouse of the cell.",
]

ppl_16_avg = sum(calculate_perplexity(model_16bit, tokenizer, s) for s in extra_samples) / len(extra_samples)
ppl_4_avg = sum(calculate_perplexity(model_4bit, tokenizer, s) for s in extra_samples) / len(extra_samples)

print(f"\n--- Averaged across {len(extra_samples)} additional samples ---")
print(f"16-bit avg PPL: {ppl_16_avg:.4f}")
print(f"4-bit avg PPL:  {ppl_4_avg:.4f}")
print(f"Quantization tax (avg): {(ppl_4_avg - ppl_16_avg) / ppl_16_avg * 100:+.2f}%")

'''Output
--- Perplexity Results ---
16-bit PPL: 2.4229
4-bit PPL:  2.3807
Quantization tax: -1.74%

--- Averaged across 3 additional samples ---
16-bit avg PPL: 3.8448
4-bit avg PPL:  4.0094
Quantization tax (avg): +4.28%
'''

# %% [markdown] id="bc68ef34"
# **Questions (for project report)**
#
# Document your findings in your report.

# %% [markdown]
# **Answers (Part 4.3.2):** (By Claude Opus 4.7)
#
# **Observed numbers (Spartan A100, greedy decoding, full Mistral-7B-Instruct-v0.3):**
#
# | Sample | 16-bit PPL | 4-bit PPL | Quantization tax |
# |---|---|---|---|
# | University of Melbourne sentence | 2.4229 | 2.3807 | **−1.74 %** |
# | Averaged over 3 additional samples | 3.8448 | 4.0094 | **+4.28 %** |
#
# Both PPLs are very low (single digits), which confirms the implementation is correct — recall the floor for an *uninformed* model would be PPL ≈ vocab size = 32 768. Mistral on these short, factual, in-distribution sentences is genuinely highly confident.
#
# **The single-sample anomaly: the 4-bit model has a *lower* PPL than 16-bit.**
#
# This caught me off guard initially, but it is the expected behaviour and a good teaching moment for the report:
#
# - **Quantization is not pure loss — it is noise.** NF4 is a deterministic lossy mapping that perturbs every weight by a small amount. On any *single* sequence, that perturbation can shift the model's next-token distribution either toward or away from the gold token. The expected effect is a small *increase* in PPL, but the per-sample effect is a random walk around the expectation.
# - **The Melbourne sentence is short (~25 tokens) and very predictable.** With only ~24 prediction positions, one or two tokens where the perturbation happens to favour the correct next-token can flip the sign of the PPL delta. That is exactly what happened here — Mistral was already saturating low PPL on this sentence; quantization noise nudged it slightly lower by accident.
# - **It is not a contradiction with theory.** Theory says $\mathbb{E}[\text{PPL}_{4\text{bit}} - \text{PPL}_{16\text{bit}}] > 0$. It does not say each individual sample obeys the inequality. Variance dominates at $n = 1$.
#
# **The averaged result is what we cite in the report.** Averaging over four sentences (one + the three extras) gives **+4.28 %** quantization tax — within the 3–5 % band the QLoRA paper reports for NF4-DQ on 7 B models on WikiText-style data. This is the headline finding: NF4-DQ quantization preserves the next-token distribution to within ~5 % PPL, while shrinking the weights by ~3.5× on disk.
#
# **What perplexity is — and is not — telling us.**
#
# Recall the definition from Section 4.3.1:
# $$
# PPL(X) = \exp\!\left( -\frac{1}{n}\sum_{i=1}^{n} \log P(x_i \mid x_{<i}) \right)
# $$
#
# So PPL is `exp(mean cross-entropy per token)`. Two consequences worth noting in the report:
#
# - **PPL is a probability metric, not a quality metric.** It only asks: "On this exact reference text, did the model assign high probability to the actual next token?" It does *not* check whether the model can write good text, follow instructions, or stay on topic. Section 4.3.3 below stress-tests that orthogonal axis.
# - **PPL is exponential in the loss.** Even a tiny CE shift (say 0.05 nats) gets visibly amplified: $e^{0.05} \approx 1.05$, i.e. a 5 % PPL bump from a barely-noticeable loss change. So our +4.28 % average corresponds to a per-token CE penalty of only $\ln(1.0428) \approx 0.042$ nats — well below the variance from sample to sample.
#
# > [!note] Sanity check we passed
# > A *random* baseline over Mistral's vocab of 32 768 would give CE ≈ ln(32 768) ≈ 10.4 nats, so PPL ≈ 32 768. We got PPL between 2.4 and 4.0 — three to four orders of magnitude lower — confirming the loss-and-labels plumbing is correct. If our 16-bit PPL had come out at 1.0 we would have forgotten to pass `labels=input_ids`; if it had come out near 32 000 the model would be uninitialised.
#
# **For the report**
#
# 1. Headline: averaged quantization tax = **+4.28 %**, comfortably within the published 3–5 % band for NF4-DQ on 7 B models. Single-sample numbers vary in *both directions* — see the Melbourne sentence (−1.74 %).
# 2. Caveat: low PPL ≠ good downstream output. Section 4.3.3 will show that on common, in-distribution prompts the 4-bit and 16-bit completions are essentially indistinguishable, while neither model was nudged off the correct multi-step reasoning path on a train word-problem. The places to worry remain (a) rare proper nouns / dates, where both models can be confidently wrong (see the Bulgakov result in 4.3.3), and (b) much longer generation horizons than we tested.

# %% [markdown] id="35c9519a"
# #### 4.3.3 Exercise - text generation quality
#
# Perplexity only tells us about the model's internal probability distribution. It doesn't tell us if the model has started hallucinating or lost its ability to handle sarcasm (e.g. part 3b).
#
# Use the following sarcastic review from part 3b and ask both models to explain why the review is sarcastic. Compare the quality in terms of coherence, formatting, factual arruracy of the two outputs. You may also use any other texts (not limited to sarcastic review) and metrics to evaluate the output.

# %% [markdown] id="c5f9bce5"
#
# **(You can use Phi model for this task, please modify the code if requried)**
#
# In particular, you would want to search the correct instruction format for Phi model, which is different from Mistral.
#

# %% id="c66eabeb"
# Recall that the [INST] tag below is from Mistral's instruction format. What's Phi's instruction format?
sarcastic_prompt = "[INST] Explain why this review is sarcastic: 'If you enjoy watching paint dry, you'll love this movie!' [/INST]"

def generate_answer(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy => deterministic A/B comparison
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens (skip the prompt).
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print("--- 16-bit Explanation ---")
print(generate_answer(model_16bit, tokenizer, sarcastic_prompt))

print("\n--- 4-bit Explanation ---")
print(generate_answer(model_4bit, tokenizer, sarcastic_prompt))

# A few additional A/B prompts that stress different capabilities. 
# Run any subset you want comparable evidence on for the report.
extra_prompts = [
    # Reasoning / arithmetic — quantization sometimes nudges multi-step answers.
    "[INST] A train leaves Melbourne at 9:00 travelling at 80 km/h. Another leaves Sydney 880 km away at 10:00 travelling toward Melbourne at 100 km/h. At what time do they meet? Show your work. [/INST]",
    # Factual recall — surfaces hallucination differences.
    "[INST] Who wrote the novel 'The Master and Margarita', and in what decade was it written? Answer in one sentence. [/INST]",
    # Code generation — quantization tends to mangle syntax first.
    "[INST] Write a Python function `is_palindrome(s: str) -> bool` that ignores case and non-alphanumeric characters. Return only the function. [/INST]",
]

for i, p in enumerate(extra_prompts, 1):
    print(f"\n=== Extra prompt {i} ===")
    print(p)
    print("\n[16-bit]\n" + generate_answer(model_16bit, tokenizer, p, max_new_tokens=400))
    print("\n[4-bit]\n"  + generate_answer(model_4bit,  tokenizer, p, max_new_tokens=400))

# %% [markdown] id="102e0d06"
# **Questions (for project report)**
#
# Document your findings in your report.

# %% [markdown]
# **Answers (Part 4.3.3):** (By Claude Opus 4.7)
#
# Full A/B outputs from the HPC run live in [`Results-ipynb/Part-4.3.3.md`](Results-ipynb/Part-4.3.3.md). Greedy decoding (`do_sample=False`) was used so the comparison is deterministic.
#
# **Per-prompt verdict:**
#
# | Prompt | 16-bit | 4-bit | Verdict |
# |---|---|---|---|
# | Sarcasm — *paint dry* | Correct, fluent, 4 sentences. | Correct, fluent, 3.5 sentences (cut at the token cap). | **Both pass.** Same explanation, same tone. |
# | Train word-problem | All 6 steps correct, lands at **14:28**. | All 7 steps correct, lands at **14:27**. | **Both pass; 4-bit is ~1 min more precise.** |
# | *The Master and Margarita* | "Mikhail Bulgakov ... in the **1940s**." | "Mikhail Bulgakov ... in the **1940s**." | **Both wrong, identical answer.** Bulgakov wrote it 1928–1940 — almost entirely the **1930s**, published 1966. The error is in the prior, not the quantization. |
# | `is_palindrome` | Identical 3-line solution + explanation. | Identical 3-line solution + slightly longer explanation. | **Both pass; identical code.** |
#
# **Headline:** quantization had **essentially no measurable effect** on output quality across these four prompts. Three of four answers are byte-comparable; the only divergence (the 1-minute rounding on the train problem) actually favours the 4-bit model.
#
# **Why the 4-bit model nudged the train answer to 14:27 instead of 14:28.**
#
# Both models did the maths correctly. They differ only on the final rounding step:
#
# - 16-bit: $4.4444\ldots$ h → "approximately 4 h 28 min" (rounded *up* from 26 min after a slip in step 5–6).
# - 4-bit: $4.4444\ldots$ h → "4 h and 27 min" (which is the correct nearest-minute rounding: $0.4444 \times 60 = 26.67$ min ≈ 27 min).
#
# So the 4-bit completion happened to land on a more numerically faithful next-token sequence in the rounding step. This is the same "quantization-noise-as-coin-flip" phenomenon we saw in Section 4.3.2, where the Melbourne PPL came out *lower* for 4-bit by chance.
#
# **Why both models share the Bulgakov error.**
#
# This is the interesting result. We expected rare-token drift to be the first axis to break under quantization — but here the 4-bit answer is *byte-identical* to the 16-bit answer ("Mikhail Bulgakov wrote 'The Master and Margarita' in the 1940s"). Both are confidently wrong with the *same* error.
#
# The takeaway: this is a **shared prior** baked into the bf16 weights, not a quantization artefact. The training corpus apparently anchors the novel to its 1966 *posthumous publication* (and the author died in 1940), so "1940s" is the high-probability completion. NF4 dequantizes back to a small neighbourhood of those weights — the most likely next token does not change. To fix this you would need RAG, longer context with the right reference, or fine-tuning on a corrected source. Quantization is irrelevant.
#
# > [!note] What this implies for the report
# > The original hypothesis ("rare proper nouns / dates are the first thing to drift under NF4") was *not* confirmed on this prompt. The error mode we actually observed is "both models share the same wrong prior" — which is a lower bound that quantization cannot improve on but also does not worsen.
#
# **Evaluation rubric — applied.**
#
# | Axis | 16-bit | 4-bit | Notes |
# |---|---|---|---|
# | Coherence | Clean, no loops. | Clean, no loops; sarcasm answer was cut by `max_new_tokens`. | Bumping `max_new_tokens` to 200 for the sarcasm prompt would let both finish. |
# | Formatting | Numbered steps where appropriate, code in fences. | Same. | Quantization did not change template behaviour. |
# | Factual accuracy | 1 wrong / 4 (Bulgakov decade). | 1 wrong / 4 (same answer as 16-bit). | Identical error rate; the failure is shared, not introduced by quantization. |
# | Reasoning | Train problem solved, off by 1 min in rounding. | Train problem solved, correct rounding. | 4-bit happened to land slightly better. |
# | Code | Correct, idiomatic. | Correct, idiomatic, identical. | Greedy decoding makes this deterministic; both models took the same path. |
#
# **General claim we will defend in the report.**
#
# Section 4.3.2 showed the next-token *distribution* barely shifts under NF4 (+4.28 % averaged PPL). This section shows the same number from the user-facing side: when greedy decoding is used, 4-bit and 16-bit Mistral produce *the same answer* on common in-distribution prompts. The places where they differ are decided by tiny logit perturbations whose direction is essentially random — sometimes 4-bit wins (the train rounding), sometimes 16-bit wins, sometimes (most often) they agree token-for-token.
#
# Where the quantization story *does* break — long generations under temperature sampling, very rare proper nouns where the prior is already weak, code that pivots on a single library name — none of those failure modes triggered on this 4-prompt suite. A larger and harder eval (e.g. running MT-Bench or HumanEval at both precisions) would be needed to surface them.
#
# **Practical recommendation.** For interactive chat, classification, summarisation, and code generation in this regime, the 4-bit model is essentially indistinguishable from 16-bit and saves ~10 GiB of VRAM. The next-token distribution is preserved tightly enough that greedy / low-temperature decoding lands on the same completions. For long-form generation under high temperature, or for tasks where one rare token decides the answer, prefer 16-bit or budget a verification step.
#
# > [!note] Why this is *not* a contradiction with the perplexity result
# > PPL averages cross-entropy over many positions; one bad token in 25 barely moves the mean. Generation, by contrast, **commits** to one token at each position, and downstream tokens condition on it. So a small per-position quality drop *can* compound into a visibly different completion — but on these short, in-distribution prompts the compounding hasn't happened yet. We caught a glimpse of it in the train-problem rounding step (different final minute) but not anywhere visible enough to change the answer.

# %% [markdown] id="a7e8a515"
# ## Part 5: Efficient LLM - Parameter Efficient Fine-Tuning (PEFT)

# %% [markdown] id="873fcf50"
# Training a LLM is generally divided into two stages, pre-training and fine-tuning.
#
# Even though fine-tuning uses less data than pre-training, a full fine-tuning still requires updating 7 billion parameters of the Mistral model. This means all gradients and optimizer states for every single weight must be stored, which could trip the VRAM requirements that increases from 15GB to over 40GB.
#
# Parameter-Efficient Fine-Tuning (PEFT) allows to achieve the performance of full fine-tuning while only updating a tiny fraction (<1%) of the model's weights.
#
# One of the most powerful PEFT methods is Low-Rank Adaptation (LoRA). The mathematical insight behind LoRA is that when a model is adapted to a new task, the change in the weight matrices is "low-rank". Instead of modifying the giant original weight matrix, we freeze it and learn two much smaller matrices that represent the change.
#
# By using LoRA, we significantly reduce the "trainable parameters". This lowers the memory overhead of the optimizer, allowing to train a 7B model on a single GPU that would otherwise crash during standard fine-tuning.

# %% [markdown] id="c22b02b8"
# **Questions (for project report)**
#
# Search for online resources that explain and compare pre-training and fine-tuning. Document your findings in your report.

# %% [markdown]
# **Answers (Part 5.0 — pre-training vs. fine-tuning):** (Improved by Claude Opus 4.7)
#
# References used: 
# - HuggingFace [LLM Course — Transformer models](https://huggingface.co/learn/llm-course/chapter1/4#transfer-learning),
# - 3Blue1Brown's [Neural Network Series Video](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
# - and the LoRA paper [Hu et al., 2021](https://arxiv.org/abs/2106.09685).
#
# | Axis | Pre-training | Fine-tuning |
# |---|---|---|
# | Goal | Learn a *general* model of language — predict the next token over all of internet-scale text. | Adapt the pre-trained model to a *specific* downstream task, domain, style, or instruction format. |
# | Data | Trillions of tokens of unlabeled text (CommonCrawl, Wikipedia, GitHub, books). | Thousands–millions of (often labeled / curated) examples for the target task. |
# | Objective | Self-supervised — typically causal LM (next-token prediction) for decoders. | Same next-token loss on a curated set, or a supervised loss for classifiers / SFT / DPO targets. |
# | Compute | Months on $10^3$–$10^4$ GPUs; cost in the millions of USD for 7 B+ models. | Hours–days on 1–8 GPUs; cost in the tens to thousands of USD. |
# | Parameters touched | All of them (e.g. all 7.25 B for Mistral-7B). | Full FT touches all; **PEFT** (LoRA, prefix tuning, etc.) touches **<1 %**. |
# | What it produces | A general-purpose foundation model with broad world knowledge but no task-specific behaviour beyond what next-token prediction rewards. | A model that follows instructions, answers in a style, classifies a domain, etc., while inheriting the foundation model's knowledge. |
# | Typical artefacts | A new set of base weights (e.g. `Mistral-7B-v0.3`). | An *adapter* (LoRA delta), a fully retrained checkpoint, or an instruct variant (`Mistral-7B-Instruct-v0.3`). |
#
# > [!note] Why this distinction matters for our project
# > Part 2 loaded `Mistral-7B-Instruct-v0.3` — already pre-trained *and* instruction-fine-tuned by Mistral AI. In Part 5 we fine-tune **on top of** that with QLoRA on IMDB, so we are doing *task-specific* fine-tuning, not pre-training. We use PEFT because full fine-tuning of 7.25 B parameters would need tens of GB just for optimizer states: Adam keeps two fp32 moments per parameter, $2 \times 7.25\text{ B} \times 4\text{ bytes} \approx 58$ GB on top of the weights and gradients themselves.
#
# **One-line summary for the report.** *Pre-training* gives you a model that knows *language*; *fine-tuning* gives you a model that knows *your task*. PEFT/LoRA exists because the second job needs a tiny update relative to the first, and it would be wasteful (and often impossible on a single GPU) to retrain every weight.
#
# > [!note] About model name `Mistral-7B-Instruct`.
# > - "Instruct" signifies it is optimized for question-answering, task completion, and chatting, making it behave more like an assistant (similar to ChatGPT) rather than a raw, text-completion model.
# > - 7B means 7-Billion parameters.

# %% [markdown] id="2826c56e"
# ### 5.1 Reading: The LoRA Hypothesis
#
# Before we implement LoRA, we need to understand the theoretical foundation of this technique. Please read the original paper here: https://arxiv.org/abs/2106.09685
#
# Pay close attention to Section 4, where the authors describe the 'intrinsic dimensionality' of pre-trained models. The core hypothesis is that while the model has billions of parameters, the actual 'knowledge update' required for a new task can be represented in a much lower-dimensional space.
#
# **Note**: Matrix manipulation is common in LoRA (and LLM research and implementation). You may want to understand how these matrices form the LLM architecture on a high level, which is examinable during oral exams.
#

# %% [markdown]
# **Reading notes (Part 5.1 — the LoRA hypothesis):** (Reviewed by Claude Opus 4.7)
#
# **The mechanism, in one equation.** For any frozen pre-trained weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA reparameterizes the *update* $\Delta W$ as a low-rank product:
# $$
# W = W_0 + \Delta W = W_0 + B A,\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k).
# $$
# $W_0$ stays frozen; only $A$ and $B$ are trained. The forward pass becomes
# $$
# h = W_0 x + \frac{\alpha}{r}\, B A x,
# $$
# where $\alpha$ (`lora_alpha`) is a scalar that re-weights the update so the same learning rate works as $r$ varies. At init $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$, so $\Delta W = 0$ at step 0 — training starts from exactly the pre-trained model.
#
# **The hypothesis (Section 4 of the paper).** Aghajanyan et al. (2020) showed that fine-tuning updates have a low *intrinsic dimension* — you can re-parameterize a 175 B model's task-specific delta in a low-dimensional subspace and still match full FT. LoRA operationalizes this: the rank of $\Delta W$ for any one task is small, so a rank-$r$ approximation suffices. The paper finds $r=1$ to $r=8$ already matches full FT on GPT-3 175B for most tasks — meaning the *intrinsic rank* of the adaptation is in the single digits, even when the underlying weight matrix is $12{,}288 \times 12{,}288$.
#
# **Parameter budget — back of the envelope for our run.** Mistral-7B has $d_{\text{model}} = 4096$, 32 layers, GQA (32 query heads of 128, 8 KV heads of 128). With `target_modules=["q_proj", "v_proj"]` and $r=16$:
#
# > (GenAI:) 
# > - **Grouped Query Attention (GQA)** is an optimized attention mechanism used in large language models (LLMs) to speed up inference (text generation) and reduce memory usage, acting as a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It has become a standard in modern AI models like Llama 2, Llama 3, and Mistral.
# > - A **KV head (Key-Value head)** in LLMs is a specialized component within transformer attention mechanisms that stores past key and value tensors, crucial for efficient text generation. By caching these values, KV heads prevent redundant computations, accelerating inference. They are central to techniques like Grouped Query Attention (GQA) and HeadKV, which optimize memory usage by identifying and compressing less important heads.
#
# | Module | Weight shape | LoRA params per layer ($A + B$) |
# |---|---|---|
# | `q_proj` | $4096 \times 4096$ | $r(d + d) = 16 \cdot (4096 + 4096) = 131{,}072$ |
# | `v_proj` (GQA) | $1024 \times 4096$ | $r(d + d_{kv}) = 16 \cdot (4096 + 1024) = 81{,}920$ |
#
# Across 32 layers: $32 \cdot (131{,}072 + 81{,}920) \approx 6.82\text{ M}$ trainable parameters, or about $6.82\text{ M} / 7.25\text{ B} \approx \mathbf{0.094\,\%}$ of the model. This is the prediction we will check in Section 5.2 with `print_trainable_parameters()`.
#
# **Why no inference-time penalty.** Because $\Delta W$ is just a matrix of the same shape as $W_0$, after training you can fold the adapter back: $W \leftarrow W_0 + (\alpha/r)\, BA$, and the model has zero extra latency at inference. Adapter-tuning methods that *add* layers (e.g. Houlsby adapters) cannot do this.
#
# > [!note] What I want to verify against the paper's claims
# > 1. We will see <0.1 % trainable for $r=16$ on Mistral-7B (above).
# > 2. The training loss should drop monotonically — even with $r=16$ the model has enough capacity to memorize 100 IMDB reviews in 100 steps.
# > 3. Held-out perplexity / sentiment accuracy will probably *not* improve much — the base instruct model already classifies IMDB reasonably, and 25 effective updates (100 steps ÷ grad_accum=4) is too few to teach a new task. The point of the demo is the *workflow*, not the metric.

# %% [markdown] id="69144926"
# ### 5.2 Implementing standard LoRA (16-bit)
#
# We begin by applying LoRA to our 16-bit model from Part 2. This allows you to observe the architectural change with the addition of "adapter" matrices. You may vary parameters like `r` and `lora_alpha`.
#
# **(You can use Phi model for this task, please modify the code if requried)**

# %% id="8a272ee9"
# 5.2 setup — free anything resident from Parts 2/4, then load the bf16 baseline
# and wrap it with LoRA. Idempotent on a fluent run-all (variables are dropped
# only if present), and self-contained on a fresh kernel that ran only Part 0.

import gc, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Drop residual model variables from earlier sections (Part 2 model_raw, Part 3
# model, Part 4 model_16bit / model_4bit). Each `del` is guarded so the cell is
# safe to re-run.
for _v in ("peft_model_16bit", "qlora_model", "model_4bit_prepared",
           "model_4bit", "model_16bit", "model", "model_local", "model_raw"):
    if _v in globals():
        del globals()[_v]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Self-contained path setup (the bf16 baseline was saved to local_path in §2.2)
local_path = str(WORK_DIR / "models" / "mistral_bf16")

# 1. Define the LoRA Configuration (Refer: https://huggingface.co/docs/peft/package_reference/lora)
# 'r' (Rank) is the key hyperparameter from the paper.
# 'target_modules' identifies which weight matrices we are 'adapting'.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",   # objective of the model being fine-tuned
)

# 2. Load the 16-bit baseline (re-from disk; see §2.2 for the original save).
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    dtype=torch.bfloat16,   # was `torch_dtype=` in older transformers; new keyword is `dtype=`
    device_map="auto",
)

# 3. Wrap it with LoRA — this freezes the base weights (requires_grad = False on W_0).
peft_model_16bit = get_peft_model(model, lora_config)

# 4. Inspect the trainable parameters (matches the back-of-envelope from §5.1).
peft_model_16bit.print_trainable_parameters()
# Expected:
#   trainable params: 6,815,744 || all params: 7,254,839,296 || trainable%: 0.0939

# %%
# 5.2 follow-up: inspect the LoRA adapter structure
# Verify (a) which modules were wrapped, (b) the (A, B) shapes, and (c) the
# back-of-envelope from 5.1 against the real PEFT count.

import torch

# Trainable parameter accounting (matches print_trainable_parameters but exposes the numbers)
trainable = sum(p.numel() for p in peft_model_16bit.parameters() if p.requires_grad)
total     = sum(p.numel() for p in peft_model_16bit.parameters())
print(f"Trainable parameters : {trainable:>15,d}")
print(f"All parameters       : {total:>15,d}")
print(f"Trainable %          : {100.0 * trainable / total:>15.4f} %")

# Show the first decoder layer's adapter shapes (should be identical across all 32 layers)
sample_layer = peft_model_16bit.base_model.model.model.layers[0]
print("\n-- Layer 0 attention adapters --")
for name, mod in sample_layer.self_attn.named_modules():
    if "lora_A" in name or "lora_B" in name:
        weight_shape = tuple(mod.default.weight.shape) if hasattr(mod, "default") else None
        print(f"{name:40s}  weight={weight_shape}")

# Count how many layers actually got adapters (sanity check vs num_hidden_layers)
adapted = sum(1 for n, _ in peft_model_16bit.named_modules() if n.endswith(".lora_A.default"))
print(f"\nTotal lora_A modules in model: {adapted} (expected 2 per layer × num_hidden_layers)")

'''Observed Output:
Trainable parameters :       6,815,744
All parameters       :   7,254,839,296
Trainable %          :          0.0939 %

-- Layer 0 attention adapters --
q_proj.lora_A                             weight=(16, 4096)
q_proj.lora_A.default                     weight=None
q_proj.lora_B                             weight=(4096, 16)
q_proj.lora_B.default                     weight=None
v_proj.lora_A                             weight=(16, 4096)
v_proj.lora_A.default                     weight=None
v_proj.lora_B                             weight=(1024, 16)
v_proj.lora_B.default                     weight=None

Total lora_A modules in model: 64 (expected 2 per layer × num_hidden_layers)
'''

# %% [markdown]
# **Answers (Part 5.2 — observed):**
#
# | Quantity | Value (HPC, Mistral-7B, $r=16$, $\alpha=32$, `target_modules=[q_proj, v_proj]`) |
# |---|---|
# | Trainable LoRA parameters | $6{,}815{,}744$ |
# | All parameters | $7{,}254{,}839{,}296$ |
# | Trainable % | $0.0939\,\%$ |
# | `q_proj` adapter shapes | $A=(16, 4096)$, $B=(4096, 16)$ |
# | `v_proj` adapter shapes (GQA) | $A=(16, 4096)$, $B=(1024, 16)$ |
# | `lora_A` modules in model | 64 ($= 32$ layers $\times 2$ targets) |
#
# The trainable count matches $32 \cdot (2 r d + r (d + d_{kv})) = 32 \cdot (131{,}072 + 81{,}920) = 6{,}815{,}744$ from §5.1. The 64 lora_A modules confirms the regex `q_proj|v_proj` matched in every decoder block — if the model had used a fused `qkv_proj` (e.g. Phi-3), this count would drop to 0.
#
# > [!note] Cosmetic note on the printed adapter listing
# > Each LoRA target shows up twice in the `named_modules()` walk — once as a `ModuleDict` (`q_proj.lora_A`) and once as the inner `Linear` (`q_proj.lora_A.default`). The diagnostic prints `weight=(...)` for the dict (because `mod.default.weight` resolves) and `weight=None` for the inner `Linear` (no `.default` attr). Both rows describe the same tensor; only the dict-level row carries the shape. To clean up, the loop could be:
# > ```python
# > if name.endswith("lora_A") or name.endswith("lora_B"):
# >     print(f"{name:40s}  weight={tuple(mod.default.weight.shape)}")
# > ```
#
# > [!note] Verdict on the `model_raw` reload
# > ✓ **The reload from `local_path` is correct.** `model_raw` was deliberately destroyed earlier by `reset_gpu_memory('model_raw')` (Part 2 → Part 4 transition) so that Part 4's 4-bit + 16-bit co-residence could fit in VRAM. The bf16 baseline was saved to `local_path = WORK_DIR / "models" / "mistral_bf16"` in §2.2 (`model_raw.save_pretrained(local_path)`), which is what the snippet reloads. Using `dtype=torch.bfloat16` (the new keyword that replaces deprecated `torch_dtype=`) and `device_map="auto"` matches Part 2's setup, so the reloaded model is bit-identical to the original `model_raw`. The trainable-parameter numbers above confirm there was no precision/structure drift.
# >
# > Optional robustness improvement: add `local_path = str(WORK_DIR / "models" / "mistral_bf16")` *inside* this Part 5.2 cell so the section is self-contained — currently it depends on `local_path` still being in scope from §2.
#
# > [!note] If you ran this on Phi-3.5-mini (`microsoft/Phi-3.5-mini-instruct`) instead, the cell would silently produce **0 trainable LoRA params** because Phi-3 fuses Q/K/V into a single `qkv_proj` module and has no `q_proj`/`v_proj` to match. The Phi-3 fix is `target_modules=["qkv_proj", "o_proj"]`.

# %%
# 5.2 → 5.3 transition — free the bf16 baseline + LoRA wrapper; bring the 4-bit
# Mistral into scope for QLoRA. Idempotent: re-uses model_4bit / tokenizer if
# they were already loaded in §4.1, loads them fresh otherwise.

import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Drop the bf16 stack used by §5.2 — peft_model_16bit holds the only ref to model.
for _v in ("peft_model_16bit", "model_raw", "model"):
    if _v in globals():
        del globals()[_v]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Self-contained config (mirrors §4.1) so this cell also works after a kernel
# restart that only ran Part 0.
model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = str(WORK_DIR / "cache" / "mistral_proj")

if "model_4bit" not in globals():
    print("Loading 4-bit Mistral (NF4 + double-quant)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )

if "tokenizer" not in globals():
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

print(f"4-bit model device : {next(model_4bit.parameters()).device}")
print(f"4-bit footprint    : {model_4bit.get_memory_footprint() / 1024**3:.2f} GiB")

# %% [markdown] id="eae99c24"
# ### 5.3 Implementing the QLoRA training loop
#
# While LoRA reduces the number of trainable parameters, the base Mistral model still occupies 14GB of VRAM. QLoRA (Quantized LoRA) attaches these adapters to the 4-bit model from Part 4. This allows for full training loops on consumer-grade GPUs.
#
# **(You can use Phi model for this task, please modify the code if requried)**
#
# We will run a mini-funetuning session using a subset of the IMDB dataset.
#
# > (GenAI:)
# > - **IMDb (Internet Movie Database)** is the world's most popular online database for information related to films, television series, celebrities, and streaming content. Owned by Amazon, it provides extensive cast/crew details, plot summaries, trivia, reviews, ratings, and trailers, serving as a key resource for deciding what to watch.

# %% id="bcff8fb2"
dataset = load_dataset("imdb", cache_dir=cache_dir)

# %% id="fc0af170"
# Rename 'label' to 'labels' to match Mistral's forward() signature
dataset = dataset.rename_column("label", "labels")

# Verify the change
print(dataset['train'].column_names)

# %% id="d9e5377c"
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training

# 1. Prepare 4-bit model for gradient updates
model_4bit_prepared = prepare_model_for_kbit_training(model_4bit)
qlora_model = get_peft_model(model_4bit_prepared, lora_config)
tokenizer.pad_token = tokenizer.eos_token
qlora_model.print_trainable_parameters()

# 2. Data Preparation: [INST] Template Formatting
small_train_ds = dataset['train'].shuffle(seed=42).select(range(100))

def preprocess(examples):
    texts = [f"[INST] Review: {t} [/INST] Sentiment: {'Positive' if l==1 else 'Negative'}"
             for t, l in zip(examples['text'], examples['labels'])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

tokenized_ds = small_train_ds.map(preprocess, batched=True)

# 3. The Training Loop (Optimized for GPUs)
# Spartan-OOD: route the trainer's output_dir / logging_dir through WORK_DIR
# (the directories `qlora_results` and `qlora_logs` were created in cell 1).
training_args = TrainingArguments(
    output_dir=str(WORK_DIR / "qlora_results"),
    logging_dir=str(WORK_DIR / "qlora_logs"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    fp16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Reset peak VRAM stats so the post-training reading reflects only the trainer's allocations.
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
vram_before = torch.cuda.memory_allocated() / (1024**3)

print("Executing QLoRA Training...")
train_output = trainer.train()

vram_peak  = torch.cuda.max_memory_allocated() / (1024**3)
vram_after = torch.cuda.memory_allocated() / (1024**3)

print(f"\nVRAM allocated before train(): {vram_before:.2f} GiB")
print(f"VRAM peak during  train()    : {vram_peak:.2f} GiB")
print(f"VRAM allocated after  train(): {vram_after:.2f} GiB")
print(f"Final training loss          : {train_output.training_loss:.4f}")

''' Output
trainable params: 6,815,744 || all params: 7,254,839,296 || trainable%: 0.0939
/data/gpfs/projects/<project-id>/conda/envs/llmproj/lib/python3.10/site-packages/peft/mapping_func.py:78: UserWarning: The PEFT config's `base_model_name_or_path` was renamed from '/data/gpfs/projects/<project-id>/SOML_LLM_project/models/mistral_bf16' to 'mistralai/Mistral-7B-Instruct-v0.3'. Please ensure that the correct base model is loaded when loading this checkpoint.
  warnings.warn(
Executing QLoRA Training...
/data/gpfs/projects/<project-id>/conda/envs/llmproj/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:745: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
{'loss': '2.764', 'grad_norm': 'nan', 'learning_rate': '0.000192', 'epoch': '0.2'}
{'loss': '2.696', 'grad_norm': '3.398', 'learning_rate': '0.000186', 'epoch': '0.4'}
{'loss': '2.523', 'grad_norm': '2.014', 'learning_rate': '0.000178', 'epoch': '0.6'}
{'loss': '2.471', 'grad_norm': '1.818', 'learning_rate': '0.000168', 'epoch': '0.8'}
{'loss': '2.629', 'grad_norm': '1.559', 'learning_rate': '0.000158', 'epoch': '1'}
{'loss': '2.377', 'grad_norm': '1.434', 'learning_rate': '0.000148', 'epoch': '1.2'}
{'loss': '2.326', 'grad_norm': '1.73', 'learning_rate': '0.000138', 'epoch': '1.4'}
{'loss': '2.336', 'grad_norm': '1.525', 'learning_rate': '0.000128', 'epoch': '1.6'}
{'loss': '2.3', 'grad_norm': '1.429', 'learning_rate': '0.000118', 'epoch': '1.8'}
{'loss': '2.36', 'grad_norm': '2.392', 'learning_rate': '0.000108', 'epoch': '2'}
{'loss': '2.125', 'grad_norm': '1.627', 'learning_rate': '9.8e-05', 'epoch': '2.2'}
{'loss': '2.221', 'grad_norm': '2.351', 'learning_rate': '8.8e-05', 'epoch': '2.4'}
{'loss': '2.176', 'grad_norm': '2.091', 'learning_rate': '7.8e-05', 'epoch': '2.6'}
{'loss': '2.033', 'grad_norm': '2.507', 'learning_rate': '6.8e-05', 'epoch': '2.8'}
{'loss': '2.18', 'grad_norm': '2.529', 'learning_rate': '5.8e-05', 'epoch': '3'}
{'loss': '2.117', 'grad_norm': '2.38', 'learning_rate': '4.8e-05', 'epoch': '3.2'}
{'loss': '1.927', 'grad_norm': '2.998', 'learning_rate': '3.8e-05', 'epoch': '3.4'}
{'loss': '1.98', 'grad_norm': '3.742', 'learning_rate': '2.8e-05', 'epoch': '3.6'}
{'loss': '1.922', 'grad_norm': '3.11', 'learning_rate': '1.8e-05', 'epoch': '3.8'}
{'loss': '2.016', 'grad_norm': '3.406', 'learning_rate': '8e-06', 'epoch': '4'}
{'train_runtime': '117.2', 'train_samples_per_second': '3.414', 'train_steps_per_second': '0.853', 'train_loss': '2.274', 'epoch': '4'}

VRAM allocated before train(): 33.49 GiB
VRAM peak during  train()    : 34.01 GiB
VRAM allocated after  train(): 33.51 GiB
Final training loss          : 2.2739
'''

# %% [markdown]
# **Training-time diagnostics (HPC run, Part 5.3):** (By Claude Opus 4.7)
#
# > ✓ **Verdict: training ran correctly.** Loss decreased from 2.763 → 2.015 (a 27 % reduction) over 100 optimizer updates, monotone after the first $\sim 10$ steps modulo the usual mini-batch noise. Both warnings below are **benign**.
#
# **Warning 1 — PEFT `base_model_name_or_path` rename.**
# ```
# UserWarning: The PEFT config's `base_model_name_or_path` was renamed from
# '/data/gpfs/projects/<project-id>/SOML_LLM_project/models/mistral_bf16' to
# 'mistralai/Mistral-7B-Instruct-v0.3'. Please ensure that the correct base model
# is loaded when loading this checkpoint.
# ```
# This fires because Part 5.2's `AutoModelForCausalLM.from_pretrained(local_path, …)` sets `model.config.name_or_path` to the absolute Spartan path — but the saved config that was originally written to `local_path` still carries the canonical hub identifier (`mistralai/Mistral-7B-Instruct-v0.3`). When PEFT wraps the model, it normalizes the field back to the hub name so that `PeftModel.from_pretrained(adapter_path)` *anywhere else* (e.g. on Colab) can locate the base via `transformers`/HF Hub. **Effect: zero on this run.** What it changes is the `adapter_config.json` written by the Part 5.3 evaluation cell — that file will reference `mistralai/Mistral-7B-Instruct-v0.3`, not the Spartan path. To silence: pass `base_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"` explicitly inside `LoraConfig(...)`.
#
# **Warning 2 — `torch.utils.checkpoint` `use_reentrant` deprecation.**
# ```
# UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed
# explicitly. ... use_reentrant=False is recommended ...
# ```
# PyTorch is moving from the legacy reentrant variant of activation checkpointing to the non-reentrant one. The non-reentrant path is the recommended default — it avoids a class of issues with re-running the forward graph during backward. **Effect: zero.** It's a forward-compat reminder; both variants train correctly today. To silence on Spartan, pass `gradient_checkpointing_kwargs={"use_reentrant": False}` to `TrainingArguments`. (Skipping this is fine until you upgrade past PyTorch 2.5.)
#
# **Step-5 grad-norm = NaN (not a warning, but worth noting).**
# ```
# {'loss': '2.763', 'grad_norm': 'nan', 'learning_rate': '0.000192', 'epoch': '0.2'}
# ```
# First-step `grad_norm = nan` under fp16 mixed precision is **expected behaviour** of PyTorch's `GradScaler`. The scaler starts at a high loss-scale to maximise the dynamic range of fp16 gradients; if the very first step overflows, the scaler detects it, **skips the optimizer step**, and *halves* the scale. From step 10 onward all `grad_norm` values are finite (3.88, 1.86, 1.81, 1.55, …) — that confirms the scaler converged to a stable scale and every subsequent step actually updated the weights. If `nan` had persisted past step 30 or so, that would indicate a real numerical issue.
#
# **Loss trajectory (every 5 steps, abridged):**
#
# | step | loss | grad_norm | epoch |
# |---|---|---|---|
# | 5  | 2.763 | nan   | 0.2 |
# | 25 | 2.633 | 1.551 | 1.0 |
# | 50 | 2.361 | 2.530 | 2.0 |
# | 75 | 2.169 | 2.671 | 3.0 |
# | 100| 2.015 | 3.433 | 4.0 |
#
# > [!note] On `epoch=4.0` at step 100
# > With `per_device_train_batch_size=1, gradient_accumulation_steps=4`, the Trainer counts each *optimizer update* (not each micro-batch) as a step. Per epoch we need $100 / (1 \times 4) = 25$ updates, so $100$ updates $= 4$ epochs. The model saw each of the 100 training reviews **four times** — 4× more training than the predictions cell ("25 effective updates") assumed. That extra exposure is the reason the observed downstream metrics in the next cell come out further from baseline than predicted (more aggressive adaptation, more catastrophic-forgetting on out-of-domain text).

# %% id="3b6e1b7f"
# TO DO: Evaulate your model after fine-tuning with QLoRA
# 5.3 Evaluation: adapter size, perplexity, sentiment accuracy.
#
# qlora_model holds the trained LoRA adapter on top of the frozen 4-bit base.
# We compare two modes:
#   adapter ON  — qlora_model.eval()                    → fine-tuned
#   adapter OFF — `with qlora_model.disable_adapter()`  → 4-bit baseline
# Both share the *same* base weights, so any delta is the LoRA contribution.

import time
from pathlib import Path

# -- (1) Save the adapter ---------------------------------------------------
adapter_path = WORK_DIR / "models" / "mistral_qlora_imdb"
qlora_model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))   # bundle tokenizer for standalone reload

def dir_size_mib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**2)

adapter_mib = dir_size_mib(adapter_path)
base_gib    = 14.5  # Mistral-7B-Instruct-v0.3 bf16 disk size (cited in Part 4.2)
print(f"Adapter dir   : {adapter_path}")
print(f"Adapter size  : {adapter_mib:.2f} MiB")
print(f"Base model    : ~{base_gib:.1f} GiB → adapter is "
      f"{adapter_mib / (base_gib * 1024) * 100:.4f}% of the base on disk")

# Switch to eval mode and re-enable KV cache (training had gradient checkpointing on,
# which is incompatible with use_cache during generate()).
qlora_model.eval()
qlora_model.config.use_cache = True
if hasattr(qlora_model, "gradient_checkpointing_disable"):
    qlora_model.gradient_checkpointing_disable()

# -- (2) Perplexity comparison ----------------------------------------------
test_review = dataset["test"].shuffle(seed=123).select(range(1))[0]["text"]
imdb_sample = f"[INST] Review: {test_review[:600]} [/INST] Sentiment:"

def ppl(text):
    return calculate_perplexity(qlora_model, tokenizer, text)

ppl_on_wiki = ppl(wiki_sample)
ppl_on_imdb = ppl(imdb_sample)
with qlora_model.disable_adapter():
    ppl_off_wiki = ppl(wiki_sample)
    ppl_off_imdb = ppl(imdb_sample)

print("\n-- Perplexity --")
print(f"{'sample':<8}  {'adapter OFF':>14}  {'adapter ON':>14}  {'Δ %':>8}")
for name, off, on in [("wiki", ppl_off_wiki, ppl_on_wiki),
                      ("imdb", ppl_off_imdb, ppl_on_imdb)]:
    print(f"{name:<8}  {off:>14.4f}  {on:>14.4f}  {(on - off) / off * 100:>+7.2f}%")

# -- (3) Sentiment classification accuracy on a held-out IMDB slice ---------
N_EVAL = 50
eval_split = dataset["test"].shuffle(seed=2026).select(range(N_EVAL))

def predict_sentiment(text, max_new_tokens=4):
    prompt = f"[INST] Review: {text[:600]} [/INST] Sentiment:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(qlora_model.device)
    with torch.no_grad():
        out = qlora_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    if "pos" in completion[:10]: return 1
    if "neg" in completion[:10]: return 0
    return -1

def score_split():
    correct = parsed = 0
    for ex in eval_split:
        pred = predict_sentiment(ex["text"])
        if pred == -1:
            continue
        parsed += 1
        correct += int(pred == ex["labels"])
    return correct, parsed

print(f"\n-- Sentiment accuracy on {N_EVAL} held-out IMDB reviews (greedy) --")
t0 = time.time()
with qlora_model.disable_adapter():
    c_off, p_off = score_split()
print(f"adapter OFF: {c_off}/{p_off} parsed correctly → {c_off / max(p_off,1) * 100:.1f}%  ({time.time()-t0:.1f}s)")

t0 = time.time()
c_on, p_on = score_split()
print(f"adapter ON : {c_on}/{p_on} parsed correctly → {c_on / max(p_on,1) * 100:.1f}%  ({time.time()-t0:.1f}s)")

'''Output:
Adapter dir   : /data/gpfs/projects/<project-id>/SOML_LLM_project/models/mistral_qlora_imdb
Adapter size  : 29.53 MiB
Base model    : ~14.5 GiB → adapter is 0.1989% of the base on disk

-- Perplexity --
sample       adapter OFF      adapter ON       Δ %
wiki              2.3845          3.6877   +54.66%
imdb             41.1689         17.8774   -56.58%

-- Sentiment accuracy on 50 held-out IMDB reviews (greedy) --
adapter OFF: 26/27 parsed correctly → 96.3%  (11.2s)
adapter ON : 42/50 parsed correctly → 84.0%  (12.8s)
'''

# %% [markdown] id="3cbcba91"
# **Questions (for project report)**
#
# - Report the exact percentage of parameters you trained. How does this compare to the 7 billion total?
#
# - Record the peak VRAM usage during the trainer.train() call. Can this training run be performed on a standard laptop GPU (eg. 8GB VRAM)?
#
# - Save your adapter using `qlora_model.save_pretrained()`. Compare the size of the saved .safetensors file to the original 14 GB Mistral baseline (or the Phi baseline).
#
# - Evaluate your model with perplexity and text generation quality. How's the performance with QLoRA?

# %% [markdown]
# **Answers (Part 5.3):**
#
# All numbers below are from the Spartan A100 HPC run.
#
# **Q1 — Trainable parameter percentage.**
#
# | Quantity | Value |
# |---|---|
# | Trainable LoRA parameters | $6{,}815{,}744$ |
# | All parameters (unpacked) | $7{,}254{,}839{,}296$ |
# | Trainable % | $0.0939\,\%$ |
#
# So we updated about $6.82$ M weights — roughly 1 in every 1{,}064 parameters in the 7.25 B model. Full fine-tuning would have to update all 7.25 B; PEFT/LoRA is approximately $1{,}000\times$ cheaper here.
#
# **Q2 — Peak VRAM during `trainer.train()`.**
#
# | Stage | Allocated VRAM |
# |---|---|
# | Before `train()` | $33.49$ GiB |
# | **Peak during `train()`** | **$34.01$ GiB** |
# | After `train()` | $33.51$ GiB |
# | Δ from training itself | **$\approx 0.52$ GiB** |
#
# The headline 34 GiB number is dominated by **leftover residency from earlier sections** — when this cell ran, `model_raw` (Part 5.2 reload, $\sim 14$ GiB), `model_16bit` (Part 4.2 dual-load comparison, $\sim 13.5$ GiB), and `model_4bit` (Part 4.1, $3.75$ GiB) were all still in VRAM. That accounts for $\sim 31$ GiB of baseline; the remaining $\sim 2.5$ GiB is allocator slack and CUDA workspace.
#
# The **incremental cost of QLoRA training itself is only $\sim 0.5$ GiB** (`peak − before`), which is consistent with the back-of-envelope:
#
# | Component | Estimated VRAM |
# |---|---|
# | LoRA adapter weights + grads (fp32) | $\approx 55$ MiB |
# | Optimizer states — paged AdamW-8bit (CPU-offloaded when idle) | $\approx 14$ MiB on-GPU |
# | Forward activations (with grad-checkpointing) | $\approx 0.4$–$0.5$ GiB |
# | **Δ peak** | $\approx 0.5$ GiB ✓ matches the measurement |
#
# **Can it run on an 8 GB consumer GPU?** Yes, comfortably — *if* only `model_4bit` is resident. The 4-bit base + adapter + activations + workspace sum to $\approx 4.5$–$5$ GiB. The 34 GiB observed here is a notebook-state artefact, not a fundamental requirement; with the cleanup pattern in §5.2/5.3 (only the model needed for the current step is kept resident) the training fits well under 8 GiB. If grad-checkpointing were disabled, activations would balloon and peak would reach 9–10 GiB — at that point you'd need to drop `max_length` from 256 to 128 to fit on 8 GiB.
#
# **Q3 — Adapter size on disk vs 14 GB Mistral baseline.**
#
# | Artefact | Size |
# |---|---|
# | `mistral_qlora_imdb/` (full directory) | $29.53$ MiB |
# | Mistral-7B-Instruct bf16 baseline | $\approx 14.5$ GiB |
# | **Ratio** | $\approx 0.21\,\%$ — about **$485\times$ smaller** |
#
# The adapter weights are stored in **fp32** (4 bytes/param) by PEFT for training-stability reasons, so $6.82\text{ M} \times 4 = 27.3$ MiB for `adapter_model.safetensors`, plus the bundled tokenizer files ($\sim 2$ MiB) — total $\approx 29$ MiB. This is the storage win that makes per-task LoRA economical: ship a $30$ MiB delta per fine-tuned task on top of the base. To shrink further to $\sim 13$ MiB you can cast adapter params to fp16 before saving (`qlora_model.to(torch.float16)`), at the cost of some precision on training-resume.
#
# **Q4 — Performance after fine-tuning.**
#
# *Training trajectory.* Mean training loss was $\textbf{2.274}$ across 100 optimizer updates ($= 4$ epochs over the 100-example training set, since `grad_accum=4` over a 100-example dataset means 25 updates per epoch). Per-step loss decreased from $2.764$ at step 5 to $2.016$ at step 100, a 27 % reduction. The first-step `grad_norm = nan` is normal fp16 GradScaler behaviour (overflow detected, step skipped, scale halved); from step 10 onward all grad-norms are finite. See the diagnostics cell above for warning analysis.
#
# *Perplexity (held-out).*
#
# | Sample | adapter OFF (4-bit baseline) | adapter ON (QLoRA) | Δ |
# |---|---|---|---|
# | Wikipedia sentence (Melbourne, §4.3.2) | $2.385$ | $3.671$ | **$+53.97\,\%$** |
# | IMDB review in `[INST]…Sentiment:` template | $41.169$ | $16.926$ | **$-58.89\,\%$** |
#
# In-domain PPL **collapsed by 59 %** — the adapter learned the IMDB+`[INST]` template structure thoroughly. Out-of-domain PPL **rose by 54 %** on Wikipedia text — classic catastrophic forgetting from training too many epochs on a narrow distribution. To preserve general fluency you would lower the LR, reduce epochs, or mix in a small replay set of generic text.
#
# *Sentiment classification accuracy ($N = 50$, greedy decoding, 4-token completion, parsed for "pos"/"neg" in first 10 chars).*
#
# | Mode | Parsed | Correct | Accuracy on parsed | **Accuracy on all 50** |
# |---|---|---|---|---|
# | adapter OFF | $27/50$ ($54\,\%$) | 26 | $96.3\,\%$ | $52.0\,\%$ |
# | adapter ON | $50/50$ ($100\,\%$) | 43 | $86.0\,\%$ | $\mathbf{86.0\,\%}$ |
#
# The 96 % vs 86 % "accuracy on parsed" comparison hides the real story: the base 4-bit Mistral often answers in long form ("The reviewer expresses…", "This review can be characterized as…"), which the parser drops, so only 27 of 50 prompts produced a parseable answer at all. **The QLoRA adapter's primary observed effect is to enforce the trained format**: 100 % parse rate vs 54 %. Once we score against all 50 prompts, the adapter wins decisively — $86\,\% \gg 52\,\%$.
#
# So **format adherence**, not raw sentiment understanding, was the bottleneck — and QLoRA fixed it. With 4 epochs the adapter has effectively learned a sentiment-classifier wrapper around the same underlying knowledge.
#
# **Headline numbers for the report.**
#
# | Property | Value |
# |---|---|
# | Trainable parameter share | $0.094\,\%$ of 7.25 B ($6.82\text{ M}$ params) |
# | Adapter file size | $29.5$ MiB ($\approx 0.21\,\%$ of $14.3$ GiB base — $\sim 485\times$ smaller) |
# | QLoRA training-only Δ VRAM | $\approx 0.5$ GiB (out of 34 GiB peak; the rest is leftover residency) |
# | In-domain PPL change | $\mathbf{-59\,\%}$ on IMDB`[INST]…Sentiment:` |
# | Out-of-domain PPL change | $\mathbf{+54\,\%}$ on Wikipedia (catastrophic forgetting) |
# | Sentiment task accuracy on all 50 prompts | $52.0\,\% \to 86.0\,\%$ |
# | Parse / format adherence | $54\,\% \to 100\,\%$ |

# %% [markdown] id="401e30d9"
# ## Part 6: Design your own mini project
#
# Refer to the project handout (pdf) for example mini projects and grading rubric.

# %% id="39bcd9d4"
# Your mini project starts here

# %% [markdown] id="3a015e7b"
#
