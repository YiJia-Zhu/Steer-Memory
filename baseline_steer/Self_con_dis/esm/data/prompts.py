from __future__ import annotations

from esm.data.tasks import TaskExample


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."


def _with_self_discover_prefix(prompt: str) -> str:
    prefix = (
        "Given this task type, select and compose a reasoning structure (steps/modules).\n"
        "Output a reusable structure as a numbered list with brief instructions per step.\n"
        "Follow this reasoning structure strictly. Solve the problem step by step.\n"
    )
    return prefix + prompt


def render_user_prompt(ex: TaskExample, template: str) -> str:
    template = template.lower()
    is_self_discover = False
    if template.endswith("_self_discover"):
        is_self_discover = True
        template = template[: -len("_self_discover")]

    def _ret(s: str) -> str:
        return _with_self_discover_prefix(s) if is_self_discover else s

    if template in {"gsm8k_0shot", "svamp_0shot"}:
        return _ret(
            "Solve the following problem step by step.\n"
            "You should show intermediate steps.\n"
            "Return the final numeric answer on the last line in the format:\n"
            "#### <answer>\n\n"
            "Problem:\n"
            f"{ex.question}\n"
        )

    if template in {"gsm8k_0shot_compact"}:
        return _ret(
            "Solve the following problem step by step.\n"
            "You should show intermediate steps.\n"
            "Keep the solution brief and avoid markdown/LaTeX.\n"
            "Return the final numeric answer on the last line in the exact format:\n"
            "#### <answer>\n\n"
            "Problem:\n"
            f"{ex.question}\n"
        )

    if template in {"math_0shot"}:
        return _ret(
            "Solve the following problem step by step.\n"
            "You should show intermediate steps.\n"
            "Put the final answer in \\boxed{<final answer>} on the last line.\n\n"
            "Problem:\n"
            f"{ex.question}\n"
        )

    if template in {"arc_0shot", "arc-c_0shot", "arc_challenge_0shot"}:
        options = ex.meta.get("options", {})
        # ARC-Challenge has {A,B,C,D} typically.
        lines = []
        for lab in ["A", "B", "C", "D", "E"]:
            if lab in options:
                lines.append(f"({lab}) {options[lab]}")
        opts_block = "\n".join(lines)
        # return (
        #     "Solve the following problem step by step.\n"
        #     "You should show intermediate steps.\n"
        #     "Choose the correct option (A, B, C, or D).\n"
        #     "Put only the chosen letter on the last line in the format:\n"
        #     "#### <A/B/C/D/E>\n\n"
        #     "Question:\n"
        #     f"{ex.question}\n\n"
        #     "Options:\n"
        #     f"{opts_block}\n"
        # )
        return _ret(
            "Solve the following problem step by step.\n"
            "You MUST explain your reasoning before giving the final choice.\n"
            "After the reasoning, choose the correct option (A, B, C, D, or E).\n"
            "Put only the chosen letter on the last line in the format:\n"
            "#### <A/B/C/D/E>\n\n"
            "Question:\n"
            f"{ex.question}\n\n"
            "Options:\n"
            f"{opts_block}\n"
        )

    if template in {"strategyqa_0shot"}:
        return _ret(
            "Solve the following problem step by step.\n"
            "You MUST show intermediate steps.\n"
            "After the reasoning, answer the question with Yes or No.\n"
            "Put only the final label on the last line in the format:\n"
            "#### <Yes/No>\n\n"
            "Question:\n"
            f"{ex.question}\n"
        )

    raise ValueError(f"未知 prompt template: {template}")


def build_chat_messages(user_prompt: str, system_prompt: str | None = DEFAULT_SYSTEM_PROMPT):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs
