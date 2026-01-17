import re
import random

from loguru import logger

# Deprecated: SYSTEM_PROMPT_MMLU was previously used to build a
# per-call system prompt in DyLAN MMLU. System behavior is now
# defined in logical agent instructions in config_mmlu*.yaml.
SYSTEM_PROMPT_MMLU = "Here's a debate. Explain your reasons at each round thoroughly.\nAll questions are single choice."

# Deprecated: ROLE_MAP was previously used to construct role-specific
# system prompts in DyLAN MMLU. Role behavior is now defined in
# logical agent instructions in config_mmlu*.yaml.
# Role maps
ROLE_MAP = {
    "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
    "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
    "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
    "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
    "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
    "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient's age, lifestyle and medical history when providing your recommendations.",
    "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
    "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
}

ACTIVATION_MAP = {"listwise": 0, "trueskill": 1, "window": 2, "none": -1}


def parse_ranks(completion, max_num=4):
    content = str(completion).strip()
    lines = content.splitlines()
    segment = content
    for line in reversed(lines):
        if line.strip():
            segment = line.strip()
            break

    pattern = r"\[?\s*(\d+)\s*,\s*(\d+)\s*\]?"
    matches = re.findall(pattern, segment)

    try:
        if not matches:
            raise ValueError("No rank pattern found")

        match = matches[-1]
        tops = [int(match[0]) - 1, int(match[1]) - 1]

        def clip(x):
            if x < 0:
                return 0
            if x > max_num - 1:
                return max_num - 1
            return x

        tops = [clip(x) for x in tops]
    except Exception:
        logger.error("error in parsing ranks; completion tail: {}", content[-200:])
        tops = random.sample(list(range(max_num)), 2)

    return tops


def parse_single_choice(reply):
    pattern = r"\(([ABCDabcd])\)"
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r"([ABCDabcd])\)"
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution


def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter
