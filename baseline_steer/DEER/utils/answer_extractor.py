import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional


def normalize_text_basic(s: str) -> str:
    return s.strip()


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+$", "", s)  # drop trailing punctuation
    return s


def normalize_number_str(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("\u2212", "-")  # unicode minus
    try:
        d = Decimal(s)
        if d == d.to_integral():
            return str(int(d))
        n = d.normalize()
        out = format(n, "f").rstrip("0").rstrip(".")
        return out if out else "0"
    except (InvalidOperation, ValueError):
        return s


_RE_LATEX_INLINE_MATH = re.compile(r"^\s*\$+(.*?)\$+\s*$", re.DOTALL)
_RE_LATEX_PARENS_MATH = re.compile(r"^\s*\\\((.*?)\\\)\s*$", re.DOTALL)
_RE_LATEX_BRACKETS_MATH = re.compile(r"^\s*\\\[(.*?)\\\]\s*$", re.DOTALL)
_RE_LATEX_INLINE_ANY = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_RE_LATEX_BRACKETS_ANY = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_RE_LATEX_DOLLAR_ANY = re.compile(r"\$(.+?)\$", re.DOTALL)
_SENTENCE_WORD_RE = re.compile(
    r"(?i)(?<!\\)\b(?!sqrt\b|frac\b|pi\b|sin\b|cos\b|tan\b|log\b|ln\b|exp\b|sec\b|csc\b|cot\b)[a-z]{3,}\b"
)

_UNIT_WORDS = {
    "unit",
    "units",
    "cm",
    "mm",
    "km",
    "inch",
    "inches",
    "ft",
    "feet",
    "yd",
    "yard",
    "yards",
    "mile",
    "miles",
    "degree",
    "degrees",
    "deg",
    "radian",
    "radians",
    "rad",
    "percent",
    "pct",
    "dollar",
    "dollars",
    "usd",
}
_UNIT_SUFFIXES_NOSPACE = sorted({u for u in _UNIT_WORDS if len(u) > 1 and u.isalpha()}, key=len, reverse=True)


def _extract_balanced_braces(s: str, open_idx: int):
    if open_idx < 0 or open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    start = None
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i], i + 1
    return None


def _strip_markdown_wrappers(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s.strip("`")
    for _ in range(2):
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            s = s[2:-2].strip()
        if s.startswith("__") and s.endswith("__") and len(s) > 4:
            s = s[2:-2].strip()
    return s.strip("*_")


def _extract_last_tex_command_arg(s: str, command: str) -> Optional[str]:
    needle = "\\" + command
    last = None
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        while k < len(s) and s[k].isspace():
            k += 1
        if k < len(s) and s[k] == "{":
            res = _extract_balanced_braces(s, k)
            if res:
                inner, end = res
                last = inner
                i = end
                continue
        i = k + 1
    return last


def _replace_tex_command_with_arg(s: str, command: str) -> str:
    needle = "\\" + command
    out = []
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            break
        out.append(s[i:j])
        k = j + len(needle)
        while k < len(s) and s[k].isspace():
            k += 1
        if k < len(s) and s[k] == "{":
            res = _extract_balanced_braces(s, k)
            if res:
                inner, end = res
                out.append(inner)
                i = end
                continue
        i = k
    out.append(s[i:])
    return "".join(out)


def _extract_last_math_segment(s: str) -> Optional[str]:
    boxed = _extract_last_tex_command_arg(s, "boxed")
    if boxed:
        return boxed
    matches = []
    for pat in (_RE_LATEX_INLINE_ANY, _RE_LATEX_BRACKETS_ANY, _RE_LATEX_DOLLAR_ANY):
        for m in pat.finditer(s):
            matches.append((m.start(), m.group(1)))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]
    return None


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _pick_last_math_token(s: str) -> str:
    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if not tokens:
        return s

    def is_math_token(tok: str) -> bool:
        if re.search(r"\d", tok):
            return True
        if "\\" in tok:
            return True
        if re.search(r"[\\^_*/()+-]", tok):
            return True
        if re.search(r"(?i)\\b(pi|sqrt|frac)\\b", tok):
            return True
        return False

    last_idx = None
    for i, tok in enumerate(tokens):
        if is_math_token(tok):
            last_idx = i

    if last_idx is None:
        return tokens[-1]

    cand = tokens[last_idx]
    if last_idx > 0:
        prev = tokens[last_idx - 1]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", prev) and re.match(r"(?:\\\\?sqrt|sqrt)\b", cand):
            cand = prev + cand
    return cand


def _pick_text_answer(s: str) -> str:
    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if not tokens:
        return s
    lower_tokens = [t.lower() for t in tokens]
    for i in range(len(tokens) - 1, -1, -1):
        if lower_tokens[i] in {"is", "are", "was", "were", "be"} and i + 1 < len(tokens):
            return tokens[i + 1]
    while tokens and tokens[0].lower() in {"the", "a", "an", "answer", "final"}:
        tokens = tokens[1:]
    return tokens[0] if tokens else s


def _is_atomic_expr(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?[\w.]+", s))


def _latex_to_ascii(s: str) -> str:
    i = 0
    while True:
        j = s.find("\\frac", i)
        if j < 0:
            break
        k = j + len("\\frac")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        num, k2 = a
        if k2 >= len(s) or s[k2] != "{":
            i = k2
            continue
        b = _extract_balanced_braces(s, k2)
        if not b:
            i = k2 + 1
            continue
        den, k3 = b
        num_s = _latex_to_ascii(num)
        den_s = _latex_to_ascii(den)
        if not _is_atomic_expr(num_s):
            num_s = f"({num_s})"
        if not _is_atomic_expr(den_s):
            den_s = f"({den_s})"
        rep = f"{num_s}/{den_s}"
        s = s[:j] + rep + s[k3:]
        i = j + len(rep)

    i = 0
    while True:
        j = s.find("\\sqrt", i)
        if j < 0:
            break
        k = j + len("\\sqrt")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        inner, k2 = a
        inner_s = _latex_to_ascii(inner)
        rep = f"sqrt({inner_s})"
        s = s[:j] + rep + s[k2:]
        i = j + len(rep)

    return s


def normalize_math_answer(s: str) -> str:
    s = s.strip()
    if not s:
        return s

    s = s.replace("\n", " ").strip()
    s = _strip_markdown_wrappers(s)
    for pat in (_RE_LATEX_INLINE_MATH, _RE_LATEX_PARENS_MATH, _RE_LATEX_BRACKETS_MATH):
        m = pat.fullmatch(s)
        if m:
            s = m.group(1).strip()
            break
    s = s.replace("\\$", "$").strip("$").strip()
    s = _replace_tex_command_with_arg(s, "text")
    s = s.replace("\\,", "").replace("\\ ", "").replace("\\!", "")
    s = _fix_fracs(s)
    s = _fix_sqrt(s)
    s = re.sub(r"\\(left|right)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    if s.count("/") == 1:
        num, den = [x.strip() for x in s.split("/")]
        if num and den and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", num) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", den):
            try:
                frac = Fraction(Decimal(num), Decimal(den))
                return f"{frac.numerator}/{frac.denominator}"
            except Exception:
                pass

    mseg = _extract_last_math_segment(s)
    if mseg:
        try:
            return normalize_math_answer(mseg)
        except Exception:
            pass

    if _SENTENCE_WORD_RE.search(s):
        return _pick_text_answer(s)

    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if tokens:
        s = tokens[-1]

    s = re.sub(r"\{\\([a-zA-Z]+)\}", r"\\\1", s)
    s = _latex_to_ascii(s)
    s = s.replace(" ", "")
    return s


_RE_HASH_ANSWER_NUM = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE | re.MULTILINE)
_RE_LAST_NUMBER = re.compile(r"([-+]?\d[\d,]*\.?\d*)")
_RE_HASH_ANSWER_LETTER = re.compile(r"####\s*([A-E])", re.IGNORECASE | re.MULTILINE)
_RE_LAST_LETTER = re.compile(r"\b([A-E])\b")
_RE_HASH_YN = re.compile(r"####\s*(yes|no)", re.IGNORECASE | re.MULTILINE)
_RE_ANSWER_LINE = re.compile(
    r"(?im)^\s*[*_`]*\s*(?:final\s+answer|answer)\s*[*_`]*\s*[:=]\s*(.+?)\s*$"
)
_RE_ANSWER_MARKER_ONLY = re.compile(r"(?im)^\s*[*_`]*\s*(?:final\s+answer|answer)\s*[*_`]*\s*[:=]?\s*[*_`]*\s*$")
_RE_ANSWER_INLINE = re.compile(r"(?i)(?:final\s+answer|answer)\s*[*_`]*\s*(?:is|=|:)\s*([^\n]+)")


def _extract_latex_braced_command_args(text: str, command: str) -> list[str]:
    needle = "\\" + command
    out = []
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        while k < len(text) and text[k].isspace():
            k += 1
        if k >= len(text) or text[k] != "{":
            i = j + len(needle)
            continue
        depth = 0
        start = None
        end = None
        for t in range(k, len(text)):
            ch = text[t]
            if ch == "{":
                depth += 1
                if depth == 1:
                    start = t + 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    end = t
                    break
            if depth == 0 and start is not None and end is not None:
                break
        if start is not None and end is not None and end >= start:
            out.append(text[start:end])
            i = end + 1
        else:
            i = k + 1
    return out


def normalize_task_name(task: str) -> str:
    return str(task).lower().strip()


def extract_pred(task: str, text: str) -> Optional[str]:
    task = normalize_task_name(task)
    if task in {"gsm8k", "svamp", "aime", "aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        m = _RE_HASH_ANSWER_NUM.findall(text)
        if m:
            return normalize_number_str(m[-1])
        nums = _RE_LAST_NUMBER.findall(text)
        if nums:
            return normalize_number_str(nums[-1])
        return None

    if task in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa", "commonsense_qa", "commonsenseqa"}:
        m = _RE_HASH_ANSWER_LETTER.findall(text)
        if m:
            return normalize_label(m[-1]).upper()
        m2 = _RE_LAST_LETTER.findall(text)
        if m2:
            return normalize_label(m2[-1]).upper()
        return None

    if task in {"strategyqa", "strategy_qa"}:
        m = _RE_HASH_YN.findall(text)
        if m:
            return normalize_label(m[-1]).title()
        m2 = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        if m2:
            return normalize_label(m2[-1]).title()
        return None

    if task in {"math", "competition_math", "math500", "math-500"}:
        boxed = _extract_latex_braced_command_args(text, "boxed")
        if boxed:
            return normalize_math_answer(boxed[-1])
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = _RE_ANSWER_LINE.match(ln)
            if m:
                ans = m.group(1).strip()
                if ans:
                    norm = normalize_math_answer(ans)
                    if norm:
                        return norm
            if _RE_ANSWER_MARKER_ONLY.match(ln):
                for j in range(i + 1, len(lines)):
                    nxt = lines[j].strip()
                    if nxt:
                        return normalize_math_answer(nxt)
                break
        ans_inline = _RE_ANSWER_INLINE.findall(text)
        if ans_inline:
            return normalize_math_answer(ans_inline[-1])
        m = _RE_HASH_ANSWER_NUM.findall(text)
        if m:
            return normalize_math_answer(m[-1])
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if lines:
            return normalize_math_answer(lines[-1])
        return None

    return None


def extract_gold(task: str, gold_field: str) -> Optional[str]:
    task = normalize_task_name(task)
    if task == "gsm8k":
        return extract_pred("gsm8k", gold_field)
    if task == "svamp":
        return normalize_number_str(gold_field)
    if task in {"aime", "aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        return normalize_number_str(gold_field)
    if task in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa", "commonsense_qa", "commonsenseqa"}:
        return normalize_label(gold_field).upper()
    if task in {"strategyqa", "strategy_qa"}:
        return normalize_label(gold_field).title()
    if task in {"math", "competition_math", "math500", "math-500"}:
        boxed = _extract_latex_braced_command_args(gold_field, "boxed")
        if boxed:
            return normalize_math_answer(boxed[-1])
        return None
    return None
