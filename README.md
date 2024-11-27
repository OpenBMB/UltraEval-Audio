
![assets/logo.png](assets/logo.png)

# Leaderboard
> **Foundation Modal**: Audio + Limited Text (Optional) → Text
> - This modal primarily focuses on traditional audio tasks such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS).
>
> **Chat Modal**: Audio + Text → Text
> - This modal is designed for interactive applications like chatbots and voice assistants. It includes tasks such as Audio Question Answering, Music Question Answering, Medicine Classification and emotional recognition.

## Foundation Leaderboard

| rank | 任务                           | model type | avg        | asr(100-wer) | ast     |
|------|--------------------------------|------------|------------|--------------|---------|
| 1    | qwen2-audio                    | foundation | 66.69675   | 95.346       | 38.0475 |
| 2    | gemini-1.5-pro                 | chat       | 64.80675   | 94.201       | 35.4125 |
| 3    | qwen2-audio-instruction        |   chat         | 63.94425   | 93.366       | 34.5225 |
| 4    | whisper                        | foundation   | 61.20925   | 93.491       | 28.9275 |
| 5    | qwen-audio                     | foundation   | 51.58375   | 73.025       | 30.1425 |
| 6    | gpt4o-realtime                 |     chat       | 44.41400   | 61.193       | 27.6350 |
| 7    | gemini-1.5-flash               |     chat       | 38.90675   | 51.891       | 25.9225 |
| 8    | qwen-audio-chat                |     chat       | 13.14925   | 15.501       | 10.7975 |
| 9    | ultravox                       |    chat        | -107.61175 | -221.746     | 6.5225 |


## Chat Leaderboard

| rank | 领域                       | medicine | music      | sound      | speech     | score      |
|------|----------------------------|----------|------------|------------|------------|------------|
| 1    | qwen2-audio-instruction     | 30.525   | 57.563333  | 70.013333  | 67.678750  | 56.445104  |
| 2    | gemini-1.5-pro              | 54.355   | 39.276333  | 48.613333  | 65.853125  | 52.024448  |
| 3    | gemini-1.5-flash            | 35.135   | 28.440000  | 38.526667  | 58.136250  | 40.059479  |
| 4    | gpt4o-realtime              | 30.300   | 13.133333  | 26.070000  | 56.966250  | 31.617396  |
| 5    | ultravox                    | 40.935   | 3.196667   | 48.420000  | 29.971250  | 30.630729  |
| 6    | qwen-audio-chat             | 0.000    | 0.000000   | 0.000000   | 0.013750   | 0.003438   |


<table>
<tr>
<td><img src="assets/audio_foundation.png" alt="图片 1 描述"></td>
<td><img src="assets/audio_chat_leaderboard.png" alt="图片 2 描述"></td>
</tr>
</table>

# Support datasets

![assets/dataset_distribute.png](assets/dataset_distribute.png)
# Changelog🔥

- [2024/11/11] We support gpt-4o-realtime-preview-2024-10-01(use as `gpt4o_audio`)

- [2024/10/8] We support 30+ datasets!

- [2024/9/7] We support `vocalsound`, `MELD` benchmark!

- [2024/9/6] We support `Qwen/Qwen2-Audio-7B`, `Qwen/Qwen2-Audio-7B-Instruct` models!

# Overview

AudioEvals is an open-source framework designed for the evaluation of large audio models (Audio LLMs).
With this tool, you can easily evaluate any Audio LLM in one go.

Not only do we offer a ready-to-use solution that includes a collection of
audio benchmarks and evaluation methodologies, but we also provide the capability for
you to customize your evaluations.


# Quick Start

## ready env
```shell
git clone https://github.com//AduioEval.git
cd AduioEval
conda create -n aduioeval python=3.10 -y
conda activate aduioeval
pip install -r requirments.txt
```

## run
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir log/
# eval gemini model only when you are in USA
export GOOGLE_API_KEY=$your-key
python audio_evals/main.py --dataset KeSpeech-sample --model gemini-pro

# eval qwen-audio api model
export DASHSCOPE_API_KEY=$your-key
python audio_evals/main.py --dataset KeSpeech-sample --model qwen-audio

# eval qwen2-audio  offline model in local
pip install -r requirments-offline-model.txt
python audio_evals/main.py --dataset KeSpeech-sample --model qwen2-audio-offline
```

## res

After program executed, you will get the performance in console and detail result as below:

```txt
- res
    |-- $time-$name-$dataset.jsonl
```

## Performance

![assets/performance.png](assets/performance.png)


> () is offical performance


## Usage

![assets/img_1.png](assets/img_1.png)

To run the evaluation script, use the following command:

```bash
python audio_evals/main.py --dataset <dataset_name> --model <model_name>
```

## Dataset Options

The `--dataset` parameter allows you to specify which dataset to use for evaluation. The following options are available:

- `tedlium-release1`
- `tedlium-release2`
- `tedlium-release3`
- `catdog`
- `audiocaps`
- `covost2-en-ar`
- `covost2-en-ca`
- `covost2-en-cy`
- `covost2-en-de`
- `covost2-en-et`
- `covost2-en-fa`
- `covost2-en-id`
- `covost2-en-ja`
- `covost2-en-lv`
- `covost2-en-mn`
- `covost2-en-sl`
- `covost2-en-sv`
- `covost2-en-ta`
- `covost2-en-tr`
- `covost2-en-zh`
- `covost2-zh-en`
- `covost2-it-en`
- `covost2-fr-en`
- `covost2-es-en`
- `covost2-de-en`
- `GTZAN`
- `TESS`
- `nsynth`
- `meld-emo`
- `meld-sentiment`
- `clotho-aqa`
- `ravdess-emo`
- `ravdess-gender`
- `COVID-recognizer`
- `respiratory-crackles`
- `respiratory-wheezes`
- `KeSpeech`
- `audio-MNIST`
- `librispeech-test-clean`
- `librispeech-dev-clean`
- `librispeech-test-other`
- `librispeech-dev-other`
- `mls_dutch`
- `mls_french`
- `mls_german`
- `mls_italian`
- `mls_polish`
- `mls_portuguese`
- `mls_spanish`
- `heartbeat_sound`
- `vocalsound`
- `fleurs-zh`
- `voxceleb1`
- `voxceleb2`
- `chord-recognition`
- `wavcaps-audioset`
- `wavcaps-freesound`
- `wavcaps-soundbible`
- `air-foundation`
- `air-chat`
- `desed`
- `peoples-speech`
- `WenetSpeech-test-meeting`
- `WenetSpeech-test-net`
- `gigaspeech`
- `aishell-1`
- `cv-15-en`
- `cv-15-zh`
- `cv-15-fr`
- `cv-15-yue`


### support dataset detail
| <dataset_name>    | name                     | task                              | domain             | metric     |
|-------------------|--------------------------|-----------------------------------|--------------------|------------|
| tedlium-*         | tedlium                  | ASR(Automatic Speech Recognition) | speech             | wer        |
| clotho-aqa        | ClothoAQA                | AQA(AudioQA)                      | sound              | acc        |
| catdog            | catdog                   | AQA                               | sound              | acc        |
| mls-*             | multilingual_librispeech | ASR                               | speech             | wer        |
| KeSpeech          | KeSpeech                 | ASR                               | speech             | cer        |
| librispeech-*     | librispeech              | ASR                               | speech             | wer        |
| fleurs-*          | FLEURS                   | ASR                               | speech             | wer        |
| aisheel1          | AISHELL-1                | ASR                               | speech             | wer        |
| WenetSpeech-*     | WenetSpeech              | ASR                               | speech             | wer        |
| covost2-*         | covost2                  | STT(Speech Text Translation)      | speech             | BLEU       |
| GTZAN             | GTZAN                    | MQA(MusicQA)                      | music              | acc        |
| TESS              | TESS                     | EMO(emotional recognition)        | speech             | acc        |
| nsynth            | nsynth                   | MQA                               | music              | acc        |
| meld-emo          | meld                     | EMO                               | speech             | acc        |
| meld-sentiment    | meld                     | SEN(sentiment recognition)        | speech             | acc        |
| ravdess-emo       | ravdess                  | EMO                               | speech             | acc        |
| ravdess-gender    | ravdess                  | GEND(gender recognition)          | speech             | acc        |
| COVID-recognizer  | COVID                    | MedicineCls                       | medicine           | acc        |
| respiratory-*     | respiratory              | MedicineCls                       | medicine           | acc        |
| audio-MNIST       | audio-MNIST              | AQA                               | speech             | acc        |
| heartbeat_sound   | heartbeat                | MedicineCls                       | medicine           | acc        |
| vocalsound        | vocalsound               | MedicineCls                       | medicine           | acc        |
| voxceleb*         | voxceleb                 | GEND                              | speech             | acc        |
| chord-recognition | chord                    | MQA                               | music              | acc        |
| wavcaps-*         | wavcaps                  | AC(AudioCaption)                  | sound              | acc        |
| air-foundation    | AIR-BENCH                | AC,GEND,MQA,EMO                   | sound,music,speech | acc        |
| air-chat          | AIR-BENCH                | AC,GEND,MQA,EMO                   | sound,music,speech | GPT4-score |
| desed             | desed                    | AQA                               | sound              | acc        |
| peoples-speech    | peoples-speech           | ASR                               | speech             | wer        |
| gigaspeech        | gigaspeech               | ASR                               | speech             | wer        |
| cv-15-*           | common voice 15          | ASR                               | speech             | wer        |

eval your dataset: [docs/how add a dataset.md](docs%2Fhow%20add%20a%20dataset.md)


### Model Options

The `--model` parameter allows you to specify which model to use for evaluation. The following options are available:

- `qwen2-audio`: Use the Qwen2 Audio model.
- `gemini-pro`: Use the Gemini 1.5 Pro model.
- `gemini-1.5-flash`: Use the Gemini 1.5 Flash model.
- `qwen-audio`: Use the qwen2-audio-instruct Audio API model.

eval your model: [docs/how eval your model.md](docs%2Fhow%20eval%20your%20model.md)

# Contact us
If you have questions, suggestions, or feature requests regarding AudioEvals, please submit GitHub Issues to jointly build an open and transparent UltraEval evaluation community.


# Citation
