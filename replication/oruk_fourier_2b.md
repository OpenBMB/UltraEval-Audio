# Oruk-Fourier-2B Evaluation Results

**Model**: [oruk-fourier-2b](../registry/model/oruk-fourier-2b.yaml)
**Evaluation Date**: 2026/11/25
**Model Card**: [oruk/oruk-fourier-2b](https://huggingface.co/oruk/oruk-fourier-2b) (pinned revision `8c2a8871f1304aa2f634da4fb95e24438c4945b9`)
**Base Architecture**: `Qwen/Qwen3-ASR-1.7B` (`Qwen3ASRForConditionalGeneration` via native Hugging Face `transformers`)
**License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial weights)
**Hardware / Runtime**: 1x NVIDIA H200 (BF16), `torch==2.8.0+cu128`, `transformers==5.24.0.dev0`

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better; computed via `audio_evals.lib.wer.compute_wer(..., language="13a")` / `PracticeWER`)

---
**Notes**:
- `oruk/oruk-fourier-2b` is an **English-only** ASR model fine-tuned from `Qwen/Qwen3-ASR-1.7B`, so only the 7 English ASR benchmarks from [replication/qwen3_asr.md](./qwen3_asr.md) are evaluated here (Chinese and dialect benchmarks are out of scope).
- **Decoding protocol disclosure**: `oruk-fourier-2b` uses its published `generation_config.json` defaults (`num_beams=3`, `no_repeat_ngram_size=8`, `do_sample=False`, `max_new_tokens=512`, `32,253` `bad_words_ids`, `language="en"`). For transparent apples-to-apples context, we also ran a matched native BF16 greedy control (`Qwen/Qwen3-ASR-1.7B-hf`, `num_beams=1`, `do_sample=False`, `language="en"`) on the exact same H200 and dataset splits alongside the upstream `qwen3-asr-1.7b` numbers from `replication/qwen3_asr.md`.
- **Single-item worker vs. batched parity**: Full 7-dataset inference was executed in BF16 (`batch_size=32` for `oruk-fourier-2b`, `batch_size=128` for `Qwen/Qwen3-ASR-1.7B-hf`) and scored with unmodified UltraEval-Audio `ContentExtract` + `PracticeWER(lang="13a")`. A live GPU parity check comparing the isolated single-item worker (`audio_evals/lib/OrukFourier/main.py`) against batched inference on the first 16 clips of `librispeech-test-clean` matched on **16/16** transcripts (`abs_wer_diff = 0.0`).
- **Checkpoint integrity**: `model.safetensors` (`4,076,193,080` bytes, SHA-256 `e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c`) and `tokenizer.json` (`11,429,377` bytes, SHA-256 `bd2a97b55c8f7f9c328c73ee9b9178771037e9f566dfca8e238a063d41cbac92`) were verified before evaluation.

## ASR (English)

| task | dataset | measure | oruk-fourier-2b (3-beam) | Qwen3-ASR-1.7B-hf (paired greedy) | qwen3-asr-1.7b ([ref](./qwen3_asr.md)) | eval_cli | note |
|------|---------|---------|--------------------------:|----------------------------------:|---------------------------------------:|----------|------|
| asr(en) | librispeech-test-clean | wer⬇️ | **1.41** | 1.65 | 1.62 | [1] | n=2,619; 196.6x RTFx (base: 644.2x) |
| asr(en) | librispeech-test-other | wer⬇️ | **2.60** | 3.37 | 3.39 | [2] | n=2,939; 182.3x RTFx (base: 611.4x) |
| asr(en) | librispeech-dev-clean | wer⬇️ | **1.57** | 1.61 | 1.63 | [3] | n=2,703; 193.0x RTFx (base: 638.6x) |
| asr(en) | librispeech-dev-other | wer⬇️ | **3.03** | 3.09 | 3.06 | [4] | n=2,864; 182.7x RTFx (base: 615.8x) |
| asr(en) | gigaspeech | wer⬇️ | **7.54** | 8.29 | 8.29 | [5] | n=19,870 non-empty after `asr_text_post_processing`; 0.0% fail_rate; 168.3x RTFx (base: 541.1x) |
| asr(en) | cv-15-en | wer⬇️ | 7.34 | **7.26** | 7.34 | [6] | n=16,386; 245.0x RTFx (base: 714.1x) |
| asr(en) | tedlium-release1 | wer⬇️ | 2.33 | **2.27** | 2.31 | [7] | n=1,155; 173.8x RTFx (base: 526.3x) |
| **avg** | **7-dataset macro avg** | wer⬇️ | **3.69** | 3.93 | 3.95 | | `-0.25` pp (`-6.2%` rel) vs. paired base; `-0.26` pp vs. `qwen3_asr.md` |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset librispeech-test-clean --model oruk-fourier-2b --prompt simple-asr`
[2] `python audio_evals/main.py --dataset librispeech-test-other --model oruk-fourier-2b --prompt simple-asr`
[3] `python audio_evals/main.py --dataset librispeech-dev-clean --model oruk-fourier-2b --prompt simple-asr`
[4] `python audio_evals/main.py --dataset librispeech-dev-other --model oruk-fourier-2b --prompt simple-asr`
[5] `python audio_evals/main.py --dataset gigaspeech --model oruk-fourier-2b --prompt simple-asr`
[6] `python audio_evals/main.py --dataset cv-15-en --model oruk-fourier-2b --prompt simple-asr`
[7] `python audio_evals/main.py --dataset tedlium-release1 --model oruk-fourier-2b --prompt simple-asr`
