from audio_evals.dataset.huggingface import Huggingface


class SeedTTSEval(Huggingface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_audio_hf_dataset(
        self, name, subset=None, split="", local_path="", col_aliases=None
    ):
        if col_aliases is None:
            col_aliases = {}
        if local_path:
            ds = load_from_disk(local_path)
        else:
            load_args = {"path": name}
            if subset:
                load_args["name"] = subset
            if split:
                load_args["split"] = split
            try:
                ds = load_dataset(**load_args, trust_remote_code=True)
            except Exception as e:
                logger.error(f"load args is {load_args}load dataset error: {e}")
                raise e

        for k, v in col_aliases.items():
            if v in ds.column_names:
                raise ValueError(f"col_aliases conflict with existing column name: {v}")
            ds = ds.rename_column(k, v)

        def conv2ds(ds):
            save_path = f"raw/{name}/"
            if subset:
                save_path += f"{subset}/"
            if split:
                save_path += f"{split}/"

            os.makedirs(save_path, exist_ok=True)
            # save prompt audio to
            res = list(
                save_audio_to_local(
                    ds, save_path, audio_col="prompt_audio", save_col="WavPath"
                )
            )
            res_with_ans = list(
                save_audio_to_local(
                    ds,
                    os.path.join(save_path, "ans"),
                    audio_col="ans_audio",
                    save_col="ans",
                )
            )
            for i in range(len(res)):
                res[i]["ans"] = res_with_ans[i]["ans"]
            return res

        if isinstance(ds, DatasetDict):
            result = []
            for k in ds:
                reload_ds = {
                    "name": name,
                    "subset": subset,
                    "split": k,
                    "local_path": local_path,
                    "col_aliases": col_aliases,
                }
                result.extend(self.load_audio_hf_dataset(**reload_ds))
            return result
        return conv2ds(ds)
