"""
Prompt templates for each compression level.
Each function returns the system + user message pair for the teacher model.
"""


def _system():
    return (
        "You are a math-solving assistant. Follow the user's formatting rules "
        "exactly. Always end your response with #### followed by the final "
        "numeric answer (no commas, no units)."
    )


# ── Level 0: Verbose (full natural language) ─────────────────────────────────

def level_0(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": (
            f"Solve this math problem step by step in full, clear English sentences. "
            f"Show every intermediate calculation and explain your reasoning.\n\n"
            f"Problem: {question}\n\n"
            f"End with #### [Final Number]"
        )},
    ]


# ── Level 1: Concise (bullet points, no filler) ─────────────────────────────

def level_1(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": (
            f"Solve this math problem using concise bullet points. "
            f"No filler words, no full sentences. Just the key steps and numbers.\n\n"
            f"Problem: {question}\n\n"
            f"End with #### [Final Number]"
        )},
    ]


# ── Level 2: Symbolic (equations and variable assignments) ───────────────────

def level_2(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": (
            f"Solve this math problem using ONLY mathematical notation. "
            f"Use variable assignments (e.g., x = 5, y = x * 3) and equations. "
            f"No English words except variable names. Show each step as an equation.\n\n"
            f"Problem: {question}\n\n"
            f"End with #### [Final Number]"
        )},
    ]


# ── Level 3: Shorthand / "Alien" notation ───────────────────────────────────

def level_3(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": (
            f"Solve this math problem using an ultra-compact shorthand.\n"
            f"Rules:\n"
            f"  - Use single-letter codes for entities (A, B, C...).\n"
            f"  - Use operator symbols: + - * / = -> for 'therefore'.\n"
            f"  - No English words at all.\n"
            f"  - Compress as much as possible while keeping the logic sound.\n"
            f"  - Example format: A=5; B=3*A; C=B-2 -> C=13\n\n"
            f"Problem: {question}\n\n"
            f"End with #### [Final Number]"
        )},
    ]


# ── Level 4: Extreme compression ────────────────────────────────────────────

def level_4(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _system()},
        {"role": "user", "content": (
            f"Solve this math problem using the ABSOLUTE MINIMUM number of tokens.\n"
            f"Rules:\n"
            f"  - Your ENTIRE reasoning must fit in 15 tokens or fewer.\n"
            f"  - Use single characters, digits, and symbols only.\n"
            f"  - Merge steps. Chain calculations inline.\n"
            f"  - The logic must still be correct.\n"
            f"  - Example: 5*3-2=13\n\n"
            f"Problem: {question}\n\n"
            f"End with #### [Final Number]"
        )},
    ]


# ── Dispatch table ──────────────────────────────────────────────────────────

LEVEL_PROMPT_FN = {
    0: level_0,
    1: level_1,
    2: level_2,
    3: level_3,
    4: level_4,
}


def build_prompt(level: int, question: str) -> list[dict]:
    """Return the chat messages for a given compression level and question."""
    return LEVEL_PROMPT_FN[level](question)
