# how add a dataset in AudioEvals?


In practice, you may need eval your custom audio dataset.

before this, you need now how launch a custom eval task: [how launch a custom eval task.md](how%20launch%20a%20custom%20eval%20task.md)

here are steps:


## JSON file:

### register the dataset
1. make sure your dataset file is `jsonl` format and with `WavPath` column which specific the audio file path.
2. new a file `**.yaml` in `registry/dataset/`
    content like :
    ```yaml
   $name:  # name after cli: --dataset $name
   class: audio_evals.dataset.dataset.JsonlFile
   args:
     default_task: alei_asr  # you should specify an eval task as default, you can find valid task in  `registry/eval_task`
     f_name:  # the file name
     ref_col:  # the reference answer column name in file
    ```
after registry dataset, you can eval your dataset with --dataset $name, enjoy 😘

Example:

1. create a file `my_dataset.jsonl` with `WavPath` and `Transcript` columns, the content like this:
```json lines
{"WavPath": "path/to/audio1.wav", "Transcript": "this is the first audio"}
{"WavPath": "path/to/audio2.wav", "Transcript": "this is the second audio"}
```

2. create a file `my_dataset.yaml` in `registry/dataset/` with content:
```yaml
my_dataset:
  class: audio_evals.dataset.dataset.JsonlFile
  args:
    default_task: asr
    f_name: my_dataset.jsonl     # the file name
    ref_col: Transcript           # the reference answer column name in file
```

3. eval your dataset with `--dataset my_dataset`

```sh
export PYTHONPATH=$PWD:$PYTHONPATH
export OPENAI_API_KEY=$your-key
python audio_evals/main.py --dataset my_dataset --model gpt4o_audio
```
