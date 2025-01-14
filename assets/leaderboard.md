
# Benchmarks in Leaderboard


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
| Metric | Dataset-Split          | GPT-4o-Realtime | Gemini-1.5-Pro | Gemini-1.5-Flash | Qwen2-Audio-Instruction | Qwen-Audio-Chat | MiniCPM-o 2.6 |
|:-------|:-----------------------|----------------:|---------------:|-----------------:|------------------------:|----------------:|--------------:|
| CER↓   | AIshell-1              |             7.3 |            4.5 |                9 |                     2.6 |           227.6 |           1.6 |
| CER↓   | Fleurs-zh              |             5.4 |            5.9 |             85.9 |                     6.9 |            80.2 |           4.4 |
| CER↓   | WenetSpeech-test-net)  |            28.9 |           14.3 |            279.9 |                    10.3 |          227.84 |           6.9 |
| WER↓   | librispeech-test-clean |             2.6 |            2.9 |             21.9 |                     3.1 |              54 |           1.7 |
| WER↓   | librispeech-test-other |             5.5 |            4.9 |             16.3 |                     5.7 |            62.3 |           4.4 |
| WER↓   | librispeech-dev-clean  |             2.3 |            2.6 |              5.9 |                     2.9 |            53.9 |           1.6 |
| WER↓   | librispeech-dev-other  |             5.6 |            4.4 |              7.2 |                     5.5 |            61.9 |           3.4 |
| WER↓   | Gigaspeech             |            12.9 |           10.6 |             24.7 |                     9.7 |              62 |           8.7 |
| WER↓   | Tedlium                |             4.8 |              3 |              6.9 |                     5.9 |            40.5 |             3 |
| BLEU↑  | covost2-en2zh          |            37.1 |           47.3 |             33.4 |                    39.5 |            15.7 |          48.2 |
| BLEU↑  | covost2-zh2en          |            15.7 |           22.6 |              8.2 |                    22.9 |              10 |          27.2 |


# Speech Generation Model Performance

| Metric            | Dataset              |   GPT-4o-Realtime |   GLM-4-Voice |   Mini-Omni |   Llama-Omni |   Moshi |   MiniCPM-o 2.6 |
|:------------------|:---------------------|------------------:|--------------:|------------:|-------------:|--------:|----------------:|
| ACC↑              | LlamaQuestion        |              71.7 |          50   |        22   |         45.3 |    43.7 |            61   |
| ACC↑              | Speech Web Questions |              51.6 |          32   |        12.8 |         22.9 |    23.8 |            40   |
| ACC↑              | Speech TriviaQA      |              69.7 |          36.4 |         6.9 |         10.7 |    16.7 |            40.2 |
| G-Eval(10 point)↑ | Speech AlpacaEval    |              74   |          51   |        25   |         39   |    24   |            51   |
| UTMOS↑            | AudioArena UTMOS     |               4.2 |           4.1 |         3.2 |          2.8 |     3.4 |             4.2 |
| ELO score↑        | AudioArena           |            1200   |        1035   |       897   |        875   |   865   |          1131   |
