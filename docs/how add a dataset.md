# how add a dataset in AudioEvals?


In practice, you may need eval your custom audio dataset.

before this, you need now how launch a custom eval task: [how launch a custom eval task.md](how%20launch%20a%20custom%20eval%20task.md)

here are steps:


## JSON file:

### register the dataset
1. make sure your dataset file is `jsonl` format
2. new a file `**.yaml` in `registry/dataset/`
    content like :
    ```yaml
   $name:  # name after cli: --dataset $name
   class: audio_evals.dataset.dataset.JsonlFile
   args:
     default_task: alei_asr  # you should specify an eval task as default, you can find valid task in  `registry/eval_task`
     f_name:  # the file name
     ref_col:  # the reference column name in file
    ```
after registry dataset, you can eval your dataset with --dataset $name, enjoy 😘
