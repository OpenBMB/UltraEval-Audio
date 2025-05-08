# Resume Evaluation

In practice, evaluation processes may occasionally fail due to various technical issues, such as model request network interruptions, system failures, or unexpected errors. To ensure the continuity and integrity of the evaluation, follow these steps to effectively restart and complete the process.

**Example Scenario:**


If the evaluation process for the `GPT-4o-Audio` model with the dataset `my_dataset` fails due to a model request network interruption, the last checkpoint is saved in the `res/gpt4o_audio/last_res.jsonl` file.

To restart the evaluation process, follow these steps:

```shell
python audio_evals/main.py --dataset my_dataset --model gpt4o_audio -r
```
is equivalent to:

```shell
python audio_evals/main.py --dataset my_dataset --model gpt4o_audio --resume res/gpt4o_audio/last_res.jsonl
```

This command will resume the evaluation from the last saved checkpoint, ensuring that the process continues seamlessly.
