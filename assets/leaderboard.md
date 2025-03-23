
# UltraEval-Audio Leaderboard

> update on 2025/01/13
## Audio Understanding LLM Leaderboard


| Rank | Model                   | ASR | AST |
|------|-------------------------|-----|-----|
| 🏅   | MiniCPM-o 2.6           | 96  | 38  |
| 🥈   | Gemini-1.5-Pro          | 94  | 35  |
| 🥉   | qwen2-audio-instruction | 94  | 31  |
| 4    | GPT-4o-Realtime         | 92  | 26  |
| 5    | Step-Audio-Chat         | 93  | 20  |
| 6    | Gemini-1.5-Flash        | 49  | 21  |
| 7    | Qwen-Audio-Chat         | 3   | 12  |


## Audio Generation LLM Leaderboard

| Rank | Model           | Semantic | Acoustic | AudioArena |
|------|-----------------|----------|----------|------------|
| 🏅   | GPT-4o-Realtime | 67       | 84       | 1200       |
| 🥈   | MiniCPM-o 2.6   | 48       | 80       | 1131       |
| 🥉   | GLM-4-Voice     | 42       | 82       | 1035       |
| 4    | Mini-Omni       | 16       | 64       | 897        |
| 5    | Llama-Omni      | 29       | 54       | 875        |
| 6    | Moshi           | 27       | 68       | 865        |


## Models details

> Currently, some famous audio LLMs, such as Step-1o-Audio by StepFun and ByteDance's Doubao,
> are product-level offerings that do not provide external API access.
> Additionally, Gemini-2.0-Exp is still in the experimental phase and does not offer standard API capabilities(only 2 QPM).
> As a result, these models are not included in the evaluation.

| Models                  | Organization | Open-Source | Audio Understanding | Audio Generation | Languages        | Notes (Evaluation of closed-source models may be affected by security reviews, so evaluation dates are specified)               |
|:------------------------|:-------------|:------------|:--------------------|:-----------------|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------|
| MiniCPM-o 2.6           | OpenBMB      | Yes         | ✅                   | ✅                | Chinese, English |                                                                                                                                 |
| GPT-4o-Realtime         | OpenAI       | No          | ✅                   | ✅                | Multilingual     | Model version evaluated: preview-2024-10-01. Audio Understanding evaluation date: 2024-12-23. Audio Generation date: 2024-10-29 |
| Gemini-1.5-Pro          | GOOGLE       | No          | ✅                   | ❌                | Multilingual     | Evaluation date: 2024-12-16                                                                                                     |
| Qwen2-Audio-Instruction | ALI          | Yes         | ✅                   | ❌                | Multilingual     |                                                                                                                                 |
| Gemini-1.5-Flash        | GOOGLE       | No          | ✅                   | ❌                | Multilingual     | Evaluation date: 2024-12-18                                                                                                     |
| Qwen-Audio-Chat         | ALI          | Yes         | ✅                   | ❌                | Multilingual     |                                                                                                                                 |
| GLM-4-Voice             | ZhiPu        | Yes         | ❌                   | ✅                | Chinese, English |                                                                                                                                 |
| LLama-Omni              | ITCNLP       | Yes         | ❌                   | ✅                | English          |                                                                                                                                 |
| Mini-Omni               | gpt-omni     | Yes         | ❌                   | ✅                | English          |                                                                                                                                 |
| Moshi                   | Kyutai       | Yes         | ❌                   | ✅                | English          |                                                                                                                                 |
| Step-Audio-Chat         | stepfun      | Yes         | ✅                   | ✅                | Chinese, English |                                                                                                                                 |


> [AudioArena](https://huggingface.co/spaces/openbmb/AudioArena) an open platform that enables users
> to compare the performance of speech large language models through blind testing and voting, providing a fair
> and transparent leaderboard for model

| Dataset                    | Name                       | Task                              | Domain        | metric    |
|----------------------------|----------------------------|-----------------------------------|---------------|-----------|
| speech-chatbot-alpaca-eval | speech-chatbot-alpaca-eval | Speech Semantic                   | speech2speech | GPT-score |
| llama-questions            | llama-questions            | Speech Semantic                   | speech2speech | acc       |
| speech-web-questions       | speech-web-questions       | Speech Semantic                   | speech2speech | acc       |
| speech-triviaqa            | speech-triviaqa            | Speech Semantic                   | speech2speech | acc       |
| tedlium-1                  | tedlium                    | ASR(Automatic Speech Recognition) | speech        | wer       |
| librispeech-test-clean     | librispeech                | ASR                               | speech        | wer       |
| librispeech-test-other     | librispeech                | ASR                               | speech        | wer       |
| librispeech-dev-clean      | librispeech                | ASR                               | speech        | wer       |
| librispeech-dev-other      | librispeech                | ASR                               | speech        | wer       |
| fleurs-zh                  | FLEURS                     | ASR                               | speech        | cer       |
| aisheel1                   | AISHELL-1                  | ASR                               | speech        | cer       |
| WenetSpeech-test-net       | WenetSpeech                | ASR                               | speech        | cer       |
| gigaspeech                 | gigaspeech                 | ASR                               | speech        | wer       |
| covost2-zh2en              | covost2                    | STT(Speech Text Translation)      | speech        | BLEU      |
| covost2-en2zh              | covost2                    | STT(Speech Text Translation)      | speech        | BLEU      |
| AudioArena                 | AudioArena                 | SpeechQA                          | speech2speech | elo score |
| AudioArena UTMOS           | AudioArena UTMOS           | Speech Acoustic                   | speech2speech | UTMOS     |


#  Audio Understanding Model Performance
| Metric | Dataset-Split          | GPT-4o-Realtime | Gemini-1.5-Pro | Gemini-1.5-Flash | Qwen2-Audio-Instruction | Qwen-Audio-Chat | MiniCPM-o 2.6 | Step-Audio-Chat |
|:-------|:-----------------------|----------------:|---------------:|-----------------:|------------------------:|----------------:|--------------:|-----------------|
| CER↓   | AIshell-1              |             7.3 |            4.5 |                9 |                     2.6 |           227.6 |           1.6 | 2.1             |
| CER↓   | Fleurs-zh              |             5.4 |            5.9 |             85.9 |                     6.9 |            80.2 |           4.4 | 6.0             |
| CER↓   | WenetSpeech-test-net)  |            28.9 |           14.3 |            279.9 |                    10.3 |          227.84 |           6.9 | 10.3            |
| WER↓   | librispeech-test-clean |             2.6 |            2.9 |             21.9 |                     3.1 |              54 |           1.7 | 3.11            |
| WER↓   | librispeech-test-other |             5.5 |            4.9 |             16.3 |                     5.7 |            62.3 |           4.4 | 8.4             |
| WER↓   | librispeech-dev-clean  |             2.3 |            2.6 |              5.9 |                     2.9 |            53.9 |           1.6 | 3.0             |
| WER↓   | librispeech-dev-other  |             5.6 |            4.4 |              7.2 |                     5.5 |            61.9 |           3.4 | 9.8             |
| WER↓   | Gigaspeech             |            12.9 |           10.6 |             24.7 |                     9.7 |              62 |           8.7 | 16.72           |
| WER↓   | Tedlium                |             4.8 |              3 |              6.9 |                     5.9 |            40.5 |             3 | 6.28            |
| BLEU↑  | covost2-en2zh          |            37.1 |           47.3 |             33.4 |                    39.5 |            15.7 |          48.2 | 26.45           |
| BLEU↑  | covost2-zh2en          |            15.7 |           22.6 |              8.2 |                    22.9 |              10 |          27.2 | 25.97           |


# Speech Generation Model Performance

| Metric            | Dataset              |   GPT-4o-Realtime |   GLM-4-Voice |   Mini-Omni |   Llama-Omni |   Moshi |   MiniCPM-o 2.6 | Step-Audio-Chat |
|:------------------|:---------------------|------------------:|--------------:|------------:|-------------:|--------:|----------------:|-----------------|
| ACC↑              | LlamaQuestion        |              71.7 |            50 |        22   |         45.3 |    43.7 |            61   | 56.7            |
| ACC↑              | Speech Web Questions |              51.6 |          32   |        12.8 |         22.9 |    23.8 |            40   | 56.1            |
| ACC↑              | Speech TriviaQA      |              69.7 |          36.4 |         6.9 |         10.7 |    16.7 |            40.2 | 36.7            |
| G-Eval(10 point)↑ | Speech AlpacaEval    |              74   |          51   |          25 |         39   |    24   |            51   | 43              |
| UTMOS↑            | AudioArena UTMOS     |               4.2 |           4.1 |         3.2 |          2.8 |     3.4 |             4.2 | 4.4             |
| ELO score↑        | AudioArena           |            1200   |        1035   |       897   |        875   |   865   |          1131   | -               |
