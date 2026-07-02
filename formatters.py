OPTION_LABELS = ["A", "B", "C", "D"]


def format_medmcqa(example):
    question = example.get("question", "")
    options = [
        example.get("opa", ""),
        example.get("opb", ""),
        example.get("opc", ""),
        example.get("opd", ""),
    ]
    choices = "\n".join(f"{OPTION_LABELS[i]}. {o}" for i, o in enumerate(options))
    cop = example.get("cop", 0)
    answer_letter = OPTION_LABELS[int(cop)] if cop is not None else "A"
    return {
        "instruction": f"{question}\n\nOptions:\n{choices}\n\nAnswer with the letter only.",
        "response": answer_letter,
        "source": "medmcqa",
    }


FORMATTERS = {
    "medmcqa": format_medmcqa,
}
