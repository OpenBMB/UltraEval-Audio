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
the A/B randomization seed) plus `description` (the official per-category
guidance from `ALL_DESCRIPTIONS`). The prompting templates are not part of the
dataset; they live in `registry/prompt/emergent_tts_eval.yaml`.

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
# Chat-style TTS models (message list: system + user)
--prompt emergent-tts-chat-strong            # default: official strong prompting
--prompt emergent-tts-chat-default           # same wrapper, without the category description
--prompt emergent-tts-chat-strong-no-system  # empty system message (official Gemini client)
--prompt emergent-tts-chat-default-no-system

# Pure TTS APIs (raw text as input, category description as the instructions/style slot)
--prompt emergent-tts-strong-api             # description -> `instruction` (e.g. OpenAI TTS)
--prompt tts                                 # raw text only (`registry/prompt/tts.yaml`)

# TTS APIs with a single free-form prompt field (SeedAudio, qwen-audio-tts-next)
--prompt emergent-tts-strong-text-prompt     # description + text folded into `text_prompt`

# TTS APIs whose style slot is spelled `instruct` (Qwen3-TTS VoiceDesign / CustomVoice)
--prompt emergent-tts-strong-instruct        # description -> `instruct`

# Explicit win-rate evaluator
--evaluator emergent-tts-eval-gemini
```

## Prompting

The benchmark's prompting templates are prompt configs in
`registry/prompt/emergent_tts_eval.yaml`, rendered by the framework's shared
`Prompt` loader with the dataset row as Jinja context (`text`, `description`,
`category`, ...):

Chat-style clients receive a system message followed by one templated user
message. The official clients differ in the system message:

- `emergent-tts-chat-strong` (default here) / `emergent-tts-chat-default`:
  the official `SYSTEM_PROMPT_DEFAULT` system message plus
  `USER_MESSAGE_STRONG_TEMPLATE` (per-category description followed by the
  text) or `USER_MESSAGE_DEFAULT_TEMPLATE` (same wrapper without the
  description). This matches the official OpenAI chat-completions client.
- Qwen2.5-Omni needs no dedicated prompt config: use the `-no-system`
  variants below, and the adapter (`qwen2_5.py`) injects the official Qwen
  system prompt when the template carries no system message, matching
  `Qwen2_5OmniClient` (`model_clients.py`).
- `emergent-tts-chat-strong-no-system` /
  `emergent-tts-chat-default-no-system`: the user message alone, matching the
  official Gemini client (empty system instruction).

These templates are `role`/`contents` message lists, the shape chat models
consume.

Pure TTS APIs mirror the official `api_clients.py:116-121` clients: the
`instructions`/style slot receives the category description (not
`SYSTEM_PROMPT_DEFAULT`), and `input` carries the raw text:

- `emergent-tts-strong-api`: `text` plus `instruction: "{{description}}"`.
- `tts` (`registry/prompt/tts.yaml`): raw `text` only (no instruction) — the
  default mode, which is also what the official clients that pass the text
  unchanged use (Sesame1B, Orpheus, ElevenLabs, Deepgram).

TTS APIs that take a single free-form prompt instead of separate text and
instruction fields (SeedAudio, qwen-audio-tts-next) fold both into
`text_prompt`:

- `emergent-tts-strong-text-prompt`: category description followed by the
  text, in one read-aloud prompt. For the no-description mode use the generic
  `tts` prompt, which already passes the raw text alone.

Select any of the variants with `--prompt <name>`.

Note: the chat-style templates are message lists meant for chat models. Models
that take separate `text` + style/instruction parameters (for example OpenAI
TTS `instructions`) must use `emergent-tts-strong-api` (strong) or `tts`
(default), which match the official pure-TTS-API prompting.

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
| gpt-4o-mini-tts (alloy) | 54.83 (54.92) |
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
