import argparse
import logging
import os
from datetime import datetime

from audio_evals.eval_task import EvalTask
from audio_evals.recorder import Recorder
from audio_evals.registry import registry


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KeSpeech")
    parser.add_argument("--model", default="qwen-audio")
    parser.add_argument("--task", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--evaluator", default="")
    parser.add_argument("--agg", default="")
    parser.add_argument("--post_process", nargs="+", default=[])
    parser.add_argument("--save", default="")
    parser.add_argument("--registry_path", default="")
    parser.add_argument("--debug_mode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rand", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    return args


def main():
    time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = get_args()
    os.makedirs("log/", exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug_mode else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"log/app-{time_id}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if not args.save:
        os.makedirs(f"res/{args.model}/{args.dataset}", exist_ok=True)
        args.save = f"res/{args.model}/{args.dataset}/{time_id}.jsonl"
    overall_save = args.save.replace(".jsonl", "-overall.json")

    if args.registry_path:
        paths = args.registry_path.split()
        registry.add_registry_paths(paths)

    dataset = registry.get_dataset(args.dataset)
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        dataset = dataset.resume_from(args.resume)
    task_cfg = registry.get_eval_task(dataset.task_name)
    if args.task:
        task_cfg = registry.get_eval_task(args.task)

    attrs = dir(task_cfg)
    for attr in dir(args):
        if not attr.startswith("__") and attr in attrs and getattr(args, attr):
            setattr(task_cfg, attr, getattr(args, attr))
    logger.info("task cfg:\n{}".format(task_cfg))

    t = EvalTask(
        dataset=dataset,
        prompt=registry.get_prompt(task_cfg.prompt),
        predictor=registry.get_model(task_cfg.model),
        evaluator=registry.get_evaluator(task_cfg.evaluator),
        post_process=[registry.get_process(item) for item in task_cfg.post_process],
        agg=registry.get_agg(task_cfg.agg),
        recorder=Recorder(args.save),
    )
    res = t.run(args.limit, args.rand, args.workers)
    with open(overall_save, "w") as f:
        f.write(str(res[0]))
    with open(args.save, "r") as f:
        print(f.read())
    print(res[0])
    print(f"Results saved to {args.save}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
