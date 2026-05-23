# License and usage notes

This repository is a personal coursework record for [ELEN90088 System Optimisation and Machine Learning](https://handbook.unimelb.edu.au/subjects/elen90088) at the University of Melbourne (2026). It is **not** released under an open-source license. The notes below describe how it may and may not be used.

## What you may do

- Read the code, notebooks, results, and notes for your own learning.
- Cite specific files or commits when referring to my reproduction of the DoRA paper (e.g. in a write-up or comparison). Use the commit SHA + file path so the reference is durable.
- Reuse small, generic snippets (helper functions, config patterns, plot styling) in your own code, with attribution if the snippet is non-trivial.

## What you may not do

- Submit this work — in whole or in part, paraphrased or as-is — as your own coursework in any course, including future iterations of ELEN90088 or related subjects.
- Redistribute the official teaching-team materials in this repository (the project briefs and reference solutions under `Exercises/` and `Project-Description/`) beyond personal reference.

If you are currently enrolled in ELEN90088, consult your subject's academic integrity policy before referring to any material here. Copying submissions is a breach of the [University of Melbourne's Academic Integrity rules](https://academicintegrity.unimelb.edu.au/) and will be detected by the standard tooling.

## Upstream code licenses

The mini-project under [`mini-project-DoRA/`](./mini-project-DoRA/) reproduces results from the **DoRA** paper (Liu et al., *ICML 2024*; [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)). It uses HuggingFace's `peft` library (`LoraConfig(use_dora=True)`) at runtime and does not bundle the upstream reference implementation.

If you choose to clone the upstream reference repository [NVlabs/DoRA](https://github.com/NVlabs/DoRA) as instructed in [`mini-project-DoRA/DoRA_REFERENCE.md`](./mini-project-DoRA/DoRA_REFERENCE.md), the resulting `DoRA/` folder is governed by the **NVIDIA Source Code License** as distributed by NVlabs — *not* by this repository's terms. The cloned folder is gitignored here and not redistributed. Cite the paper rather than this reproduction for the DoRA method itself.

Library dependencies (PyTorch, HuggingFace `transformers` / `peft` / `accelerate` / `datasets` / `trl`, `bitsandbytes`, etc.) are governed by their own respective licenses; see each project's repository for terms.

## Datasets

- [`google/boolq`](https://huggingface.co/datasets/google/boolq) — see HuggingFace dataset card for license.
- [`commonsense_170k`](https://github.com/AGI-Edgerunners/LLM-Adapters) — sourced from the LLM-Adapters project; see upstream for terms.

Neither dataset is redistributed in this repository; both are fetched at runtime via HuggingFace.

## Contact

For questions about this repository or its contents, open an issue on the GitHub mirror.
