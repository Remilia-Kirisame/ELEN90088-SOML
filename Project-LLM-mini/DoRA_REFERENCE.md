# DoRA reference clone

- Paper: <https://arxiv.org/abs/2402.09353>
- Upstream: <https://github.com/NVlabs/DoRA>
- Reference commit (pinned for traceability): `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18`
- Local clone command (each machine):

  ```bash
  git clone https://github.com/NVlabs/DoRA Project-LLM-mini/DoRA
  ```

- The folder `DoRA/` is gitignored and not used at runtime. We use HuggingFace
  PEFT's `LoraConfig(use_dora=True)` instead of DoRA's bundled custom PEFT.
- Method: weight-decomposed low-rank adaptation. Decomposes a pretrained weight
  $W$ into magnitude $m$ and direction $V$; LoRA-style low-rank update on $V$,
  separate magnitude parameter on $m$.
