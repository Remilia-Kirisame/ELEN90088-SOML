---
title: "Project 2: Hands-on Large Language Models"
course: "ELEN90088 — System Optimisation and Machine Learning"
parent: "[[SOML_Project_26S1]]"
---

# Project 2: Hands-on Large Language Models

## Introduction

In this project, you will transition from a consumer of AI to an engineer who understands the mechanics of Large Language Models (LLMs). Specifically, you will explore how these models are loaded, deployed, and optimized.

This project focuses on implementation and engineering. While the theories behind the transformer architecture (Attention, LayerNorm etc.) may be covered in lectures, you will not be asked to implement these components from scratch. Instead, you will learn to use the HuggingFace library to load, use and evaluate LLMs.

This project description provides a brief outline of the tasks and expectations, so that you can compare it with other project options.

## Resources

- This project description.
- The Jupyter Notebook (`SOML_LLM_project.ipynb`) with more detailed instructions and code. You will complete your work in the notebook, and document your findings in your project report. This file is on Canvas.
- Online resources, including Generative AI: We encourage you to use Generative AI to help you learn, find help, and debug. Please make clear references if you use online articles, and/or any AI tools.

## (Please read) Preparation: Environment Setup

Working with LLMs on "specific hardware" can be much more efficient ($10\times$ to $100\times$ faster) than running LLM code on CPUs.

The easiest access to "specific hardware" is to use Google Colab. By uploading `SOML_LLM_project.ipynb` to Colab, you can immediately start with the project work without any environment setup. We have tested that the GPU/TPUs available on the free tier of Google Colab are sufficient to complete the project tasks. We discuss more about the limitations in the next section: Project structure.

Alternatively, you can work on your local device. We assume most students have either Windows or Mac OS devices for this project, and provide a `.yaml` of each device type for you to set up a local project conda environment. You can find the `.yaml` files can on Canvas.

Note that using these `.yaml` files is *optional*. SOML teaching team created the `.yaml` files so that the environment should work on most devices, which means the libraries/dependencies are not the latest version. For example, you will see `pytorch` version 2.1.0 with `pytorch-cuda` version 11.8 in the `.yaml` files, while their latest versions are 2.10.0 and 12.4 on conda respectively. If you prefer to install your own environment with more recent versions, please feel free to do so as long as you can run `SOML_LLM_project.ipynb` with your own environment.

We briefly compare Windows and Mac OS compatibility for this project, and clarify the `.yaml` files below.

- **Windows laptop/desktop with a Nvidia GPU:**
  You should be able to run *all* tasks in `SOML_LLM_project.ipynb`. As a first step, please install the project environment using `SOML_LLM_project_Windows.yaml`, or set up your own environment.
- **Apple products with M chips (e.g. recent Macbook Pro):**
  You should be able to run all tasks in `SOML_LLM_project.ipynb`, **except** part 4 quantization. For details, please refer to the Project structure section below. As a first step, please install the project environment using `SOML_LLM_project_MacOS.yaml`, or set up your own environment.
- Talk to demonstrators if you have another type of device.

Once you have set up your environment, we recommend to run the first few cells in `SOML_LLM_project.ipynb`. These cells are designed to check whether your environment can be loaded as expected, and whether the library versions are compatible with the tasks.

*Important*: During the week 7 workshop, we highly recommend you to:

- either load `SOML_LLM_project.ipynb` to Colab and confirm you can run the first few cells, part 1 and 2, and/or
- set up an environment on your local device, and confirm you can run the first few cells, part 1 and 2 in `SOML_LLM_project.ipynb`.

and raise any questions during week 7 so you are ready to work on the project tasks in the next few weeks.

You may wonder if there are other GPU options: SOML teaching tean and the university IT team are working hard to set up a project environment so that students can access more advanced GPUs. We will keep you updated. Until then, all students will be using their own available resources (Colab and/or local device) for this project.

## Project structure

This project consists of two sections:

- The first section includes part 1-5, where you are given most of the code. Your task is to understand the context in `SOML_LLM_project.ipynb`, and run the code provided to you. You will also modify and complete some code where required. In addition, there are discussion questions that help you prepare your project report.
- The second section is a mini project (part 6), where you will apply your knowledge to implement/train/evaluate tasks that are relevant to LLM.

Below is an overview of each part.

### Part 1: The pipeline

This part lets you try LLM functionalities with just 1-2 lines of code (pipeline). You will see tasks like text generation and sentiment analysis. You will also use 1-2 lines to implement other LLM tasks.

### Part 2: Behind the pipeline

This part introduces the key elements of LLM pipeline, including tokenizers, models, and datasets. You will learn to load and manage tokenizers and models.

### Part 3: Inference

You will prepare instructions, process instructions with a tokenizer, and pass tokens to LLM to complete tasks. Essentially you will understand and implement how the pipelines from part 1 actually work behind the scenes.

### Part 4: Efficient LLM - Model Compression

You will learn about quantization, and implement simple quantization methods with existing Python libraries. You will also evaluate LLMs performance with multiple metrics. Note that Mac OS may **not** support quantization. If you plan to choose this project with a Mac computer, please consider using Colab which can run quantization and save you from installing a local environment.

### Part 5: Efficient LLM - Parameter-efficient Fine-Tuning (PEFT)

You will learn to tune a (very) small part of LLM parameters so that the LLM can achieve better performance on a specific task. You will evaluate performance after tuning with multiple metrics (again).

### Part 6: Mini project

After completing part 1-5, you will have a good foundation of LLM implementation. Now it's your turn to use all these tools on a topic of interest.

We provide some example ideas below and their *max* possible grades.

- **Example 1 (borderline pass):** Tuning LoRA/QLoRA hyperparameters on Mistral/Phi and evaluate the performance. This is essentially an extension of part 5 so the maximum grade would be 'pass' if you chose this as your mini project, provided that you completed all part 1-5 and understand what happened.
- **Example 2 (pass to H3):** Multi-model (Mistral, Phi, GPT etc.) benchmarking on specific tasks. Evaluate these models on different datasets (finance, medicine, maths etc).
- **Example 3 (H3 to possible H1):** Find, implement and evaluate other efficient LLM algorithms. Examples include advanced quantization, pruning, and distillation. You will need to be able to explain the intuition of your methods, the performance and their limitations.
- **Example 4 (Impressive):** You could find a research paper and reproduce their results (some authors publish their code on Github/huggingface). Feel free to share your idea with demonstrators before your implementation so that demonstrators can estimate the workload and confirm it's feasible within the project duration and the computational resource requirements. We look forward to putting your mini-project idea as an impressive example for future students.

Note: you are encouraged to try other popular models. Many of them are availble online, but you may need to create a `huggingface` account and request access to some models (eg. Llama from META). Reach out to demonstrators if you have any questions about this.

## Which models are we using?

This project code is designed to use Mistral LLM, which has a size of around 15GB. While the teaching team confirmed it works on advanced GPUs, using Mistral LLM for part 3-5 may exceed the capability of free Google Colab, and Mac MPS equipment.

The alternative model is `Phi-3.5-mini-instruct` from Microsoft. The teaching team confirmed using this model is feasible on free Google Colab and Mac MPS equipment for all project tasks (except quantization due to Mac OS constraints).

*Important*: SOML teaching team has summarized the initial workflow of each option:

- **Work on Colab.** You will be able to use `Phi-3.5-mini-instruct`, though you will need to slightly modify the code that was written for Mistral models. There are clear hints in the `SOML_LLM_project.ipynb` where you need to modify the code.
- **Set up a local environment and test if you can use Mistral models.** If successful, you can run most of the code without any modifications because the code was written to use Mistral models. However, there's a risk that you cannot use Mistral models after you set up your local environment, which would mean you need to try Colab with `Phi-3.5-mini-instruct`.

Please note that the model choice does **not** affect your project marks. So please choose a model that leads to a faster setup process for you.

## Deliverables and assessment

- Complete the Jupyter Notebook `SOML_LLM_project.ipynb`, including the code that you need to fill in, and your mini project implementation. You *don't* need to include your responses to the questions in the Jupyter Notebook.
- Project report. We recommend documenting your responses to the questions from the Jupyter Notebook, and using these responses as part of your report.
- Oral exam.

## Optional: Install the LLM project environment on your laptop/desktop

After you have installed conda, close any open terminals you might have. Then open a new terminal and run the following command:

- Create an environment with dependencies specified in `.yaml` file:
  `conda env create -f SOML_LLM_project_Windows.yaml` (or the Mac OS `.yaml`)
- Activate the new environment: `conda activate soml_llm_env`
- To make sure we are using the right environment, go to the toolbar of `SOML_LLM_project.ipynb`, click on Kernel → Change kernel, you should see and select `soml_llm_env` in the drop-down menu.
- To deactivate an active environment after you finish your work, use `conda deactivate`
