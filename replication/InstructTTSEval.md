# InstructTTSEval Evaluation Alignment

## Dataset

- **Hugging Face**: [CaasiHUANG/InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval)
- **Official repository**: [KexinHUANG19/InstructTTSEval](https://github.com/KexinHUANG19/InstructTTSEval)
- **Paper**: [InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems](https://arxiv.org/abs/2506.16381)

The Hugging Face dataset is approximately 634 MB and contains two splits:

- `en`: 1,000 English samples
- `zh`: 1,000 Chinese samples

Each sample contains an ID, the text to synthesize, three instruction types
(`APS`, `DSD`, and `RP`), and embedded 16 kHz reference audio. APS evaluates
fine-grained acoustic control, DSD evaluates descriptive style following, and RP
evaluates role-play or scenario-based style following.


### UltraEval-Audio Evaluation Example

Set the Gemini API key and evaluate a registered TTS model on the English
subset:

```bash
export GOOGLE_API_KEY="<your-gemini-api-key>"

python audio_evals/main.py \
  --dataset instruct-tts-eval-en \
  --model <registered-tts-model-name> \
  --workers 4
```

Use `instruct-tts-eval-zh` and `instruct_tts_eval_zh` for the Chinese subset.
The integration expands every source sample into one evaluation row for each
available instruction type, producing up to 3,000 evaluation rows per split.

## Background

The official InstructTTSEval evaluation uses `gemini-2.5-pro-preview-05-06`
as the judge model to determine whether each generated audio sample follows its
instruction. This preview model is no longer available, so the UltraEval-Audio
integration uses `gemini-2.5-pro` as the judge instead.

To validate the integration, we evaluated the official reference audio with the
integrated evaluator and compared the results with those reported by the
official repository.

## English Subset

| Metric | Official result | UltraEval-Audio result | Difference |
| --- | ---: | ---: | ---: |
| APS | 93.60% | 95.10% | +1.50 pp |
| DSD | 89.80% | 91.30% | +1.50 pp |
| RP | 70.00% | 71.30% | +1.30 pp |
| AVG | 84.47% | 85.90% | +1.43 pp |

## Chinese Subset

| Metric | Official result | UltraEval-Audio result | Difference |
| --- | ---: | ---: | ---: |
| APS | 92.09% | 89.80% | -2.29 pp |
| DSD | 84.00% | 84.80% | +0.80 pp |
| RP | 65.40% | 65.50% | +0.10 pp |
| AVG | 80.50% | 80.03% | -0.47 pp |


## Conclusion

The integrated results are broadly aligned with the official results. The
absolute difference is at most 2.29 percentage points for an individual metric;
the average-score differences are 1.43 percentage points on the English subset
and 0.47 percentage points on the Chinese subset.

Exact numerical agreement is not expected because the original preview judge
has been retired and the reproduction uses `gemini-2.5-pro`. The two runs also
have different valid/null sample counts. Within these constraints, the results
support the correctness of the UltraEval-Audio InstructTTSEval integration.
