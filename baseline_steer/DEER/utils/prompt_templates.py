DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."

MATH_STYLE_DATASETS = {
    "math",
    "competition_math",
    "math500",
    "math-500",
    "aime_2024",
    "aime2024",
    "aime25",
    "aime_25",
    "aime_2025",
    "amc23",
    "amc_23",
    "amc_2023",
}

CHOICE_STYLE_DATASETS = {
    "arc-c",
    "arc",
    "arc_challenge",
    "openbookqa",
    "openbook_qa",
    "commonsense_qa",
    "commonsenseqa",
}


def normalize_dataset_name(name: str) -> str:
    return str(name).lower().strip()


def render_user_prompt(example: dict, dataset: str) -> str:
    task = normalize_dataset_name(dataset)
    question = example.get("problem") or example.get("question") or ""

    if task in MATH_STYLE_DATASETS:
        return (
            "Solve the following problem step by step.\n"
            "You should show intermediate steps.\n"
            "Put the final answer in \\boxed{<final answer>} on the last line.\n\n"
            "Problem:\n"
            f"{question}\n"
        )

    if task in CHOICE_STYLE_DATASETS:
        options = example.get("options") or example.get("choices") or {}
        lines = []
        for lab in ["A", "B", "C", "D", "E"]:
            if lab in options:
                lines.append(f"({lab}) {options[lab]}")
        opts_block = "\n".join(lines)
        return (
            "Solve the following problem step by step.\n"
            "You MUST explain your reasoning before giving the final choice.\n"
            "After the reasoning, choose the correct option (A, B, C, D, or E).\n"
            "Put only the chosen letter on the last line in the format:\n"
            "#### <A/B/C/D/E>\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{opts_block}\n"
        )

    return f"{question}\n"


def build_chat_messages(example: dict, dataset: str, system_prompt: str | None = DEFAULT_SYSTEM_PROMPT):
    user_prompt = render_user_prompt(example, dataset)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def get_answer_prompt_config(dataset: str):
    task = normalize_dataset_name(dataset)
    if "gpqa" in task:
        return "\n**Final Answer**\nI believe the final answer, rather than the option, is \\boxed", [
            " }",
            "}\n",
            "}\n\n",
            "}.",
            "}.\n",
            "}\\",
            "}}",
            ")}",
            ")}.",
            ")}\n",
        ]
    if task in CHOICE_STYLE_DATASETS:
        return "\n#### ", ["\n"]
    return "\n**Final Answer**\n\\boxed", [
        " }",
        "}\n",
        "}\n\n",
        "}.",
        "}.\n",
        "}\\",
        "}}",
        ")}",
        ")}.",
        ")}\n",
    ]
