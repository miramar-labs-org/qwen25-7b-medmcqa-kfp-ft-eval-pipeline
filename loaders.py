from datasets import load_dataset

LOADERS = {
    "medmcqa": lambda: load_dataset("openlifescienceai/medmcqa", split="train")
        .map(format_medmcqa)
        .filter(lambda x: x["instruction"] != ""),
}
