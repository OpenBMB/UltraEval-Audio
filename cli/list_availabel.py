#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from audio_evals.registry import registry


def print_models(console: Console):
    """Print available models in a table format."""
    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Model Name (args)", style="yellow")

    for model_name, model_info in registry._model.items():
        class_name = model_info.get("cls", "N/A")
        model_args = model_info.get("args", {})
        model_name_arg = model_args.get("model_name", "N/A")

        table.add_row(model_name, class_name, str(model_name_arg))

    console.print(table)


def print_datasets(console: Console):
    """Print available datasets in a table format."""
    table = Table(title="Available Datasets")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Source", style="magenta")

    num = 0
    data = []
    for dataset_name, dataset_info in registry._dataset.items():
        class_name = dataset_info.get("cls", "N/A")
        args = dataset_info.get("args", {})
        task = args.get("default_task", "N/A")
        source = args.get("name", "N/A")

        table.add_row(dataset_name, class_name, task, source)
        num += 1
        data.append(
            [
                "speech",
                task,
                re.split(r"[-_]", dataset_name)[0],
            ]
        )
    table.title = "Available {} Datasets".format(num)

    console.print(table)
    df = pd.DataFrame(data, columns=["Domain", "task", "name"])
    df = df.drop_duplicates(["name"])
    df["task"] = df["task"].map(
        lambda x: {"asr": "ASR", "asr-zh": "ASR", "ast": "AST"}.get(x, x)
    )
    df.sort_values(by=["Domain", "task"], inplace=True)


def main():
    parser = argparse.ArgumentParser(description="List available models and datasets")
    parser.add_argument("--models", action="store_true", help="Show available models")
    parser.add_argument(
        "--datasets", action="store_true", help="Show available datasets"
    )
    args = parser.parse_args()

    # If no specific option is provided, show both
    if not (args.models or args.datasets):
        args.models = True
        args.datasets = True

    console = Console()

    if args.models:
        try:
            print_models(console)
        except Exception as e:
            console.print(f"[red]Error loading models: {str(e)}[/red]")

    if args.datasets:
        try:
            print_datasets(console)
        except Exception as e:
            console.print(f"[red]Error loading datasets: {str(e)}[/red]")


if __name__ == "__main__":
    main()
