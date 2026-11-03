# EmergentTTS-Eval Evaluation Alignment

## Dataset

- **Hugging Face**: [bosonai/EmergentTTS-Eval](https://huggingface.co/datasets/bosonai/EmergentTTS-Eval)
- **Official repository**: [boson-ai/EmergentTTS-Eval](https://github.com/boson-ai/EmergentTTS-Eval)
- **Paper**: [EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge](https://arxiv.org/abs/2505.23009)

The Hugging Face dataset is approximately 1.19 GB and contains 1,645 English
test cases in the `default` / `train` split. Every row provides:

- `category` — one of `Emotions`, `Paralinguistics`, `Foreign Words`,
  `Syntactic Complexity`, `Pronunciation`, `Questions`
- `text_to_synthesize` — the text to synthesize
- `evolution_depth`, `language`
- `audio` — the baseline speech produced by `gpt-4o-mini-tts` (voice `alloy`)

UltraEval-Audio renames `text_to_synthesize` to `text`, saves the embedded
baseline audio to disk as `WavPath`, and adds `id` (source row index, used as
the A/B randomization seed) plus `instruction` (the official strong prompting
guidance for the row's category).

## Metrics

The UltraEval-Audio integration reports the win-rate metric only.

**Win rate** — a Gemini judge compares the generated audio against the
`gpt-4o-mini-tts` baseline, using the official per-category evaluation
criterion. The judge returns `winner` 0/1/2 (tie/system 1/system 2) and
0-3 scores. A tie counts as 0.5 win. System order is randomized with
`numpy.random.default_rng(row_index)`, reproducing the official position-bias
control.

Judge failures are excluded from the win-rate denominator and reported as
`JudgeFailCount`, mirroring the official `judger_parsing_failed_count`.

## UltraEval-Audio Evaluation Examples

The default task uses the official strong prompting and reports win rate.

```bash
export GOOGLE_API_KEY="<your-gemini-api-key>"

python audio_evals/main.py \
  --dataset emergent-tts-eval \
  --model <registered-tts-model-name> \
  --workers 4
```

Variants override the same task from the CLI:

```bash
# Raw text only (no category guidance)
--prompt tts

# Explicit win-rate evaluator
--evaluator emergent-tts-eval-gemini
```

## Prompting

The benchmark has two prompting modes. For pure TTS APIs the official
implementation sends the raw text by default, or the category description when
`--text_speech_strong_prompting` is passed. UltraEval-Audio maps these to the
`text` and `instruction` fields of the TTS prompt:

- strong (default here): the shared `instruct-tts` prompt
  (`{text: "{{text}}", instruction: "{{instruction}}"}`), fed by the
  dataset's `instruction` column.
- raw text: the generic `tts` prompt (`{text: "{{text}}"}` in
  `registry/prompt/tts.yaml`), selected with `--prompt tts`.

Chat-style TTS APIs in the official repository additionally wrap the text in a
"generate verbatim speech" chat message. That wrapper is model-client specific
and is not applied here; use the raw-text or instruction prompt that matches
your model's native interface.

## Outputs

The aggregation reports:

| Field | Meaning |
| --- | --- |
| `Total` / `Judged` / `JudgeFailCount` | sample counts |
| `WinRate(%)` | win rate over judged samples, tie = 0.5 |
| `TieRate(%)` | share of judged samples scored as ties |
| `<Category> ...` | the same fields per category |

## Official Result Alignment

The values in parentheses are produced by the official evaluation code; the
values outside are produced by UltraEval-Audio (WinRate/%). The official
leaderboard used `gemini-2.5-pro` as the judge, which has been retired, so the
leaderboard numbers cannot be reproduced. We therefore align only the results
produced by the official evaluation code with the same inference audio and the
same judge model: both runs use `gemini-3.8-flash`, and only win rate is
compared.

| Model | WinRate/% ↑ |
| --- | ---: |
| gpt-4o-mini-tts (alloy) | 56.01 (54.92) |
| gpt-4o-mini-tts (ballad) | 55.11 (56.05) |

## Local Validation

- `tests/test_emergent_tts_eval.py` covers dataset field injection, prompt
  placeholder filling for all six categories, judge JSON extraction and
  validation, deterministic A/B ordering, win/tie/loss mapping, retry and
  failure paths, aggregation, and registry loading.
- The Hugging Face dataset schema was verified against
  `bosonai/EmergentTTS-Eval` (`category`, `text_to_synthesize`,
  `evolution_depth`, `language`, `audio`).

Cluster validation results are summarized in
[Official Result Alignment](#official-result-alignment) above.
