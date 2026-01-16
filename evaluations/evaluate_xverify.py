from loguru import logger

MAX_RESPONSE_LENGTH = 5000


def format_prompt(query, response, ground_truth):
    prompt = f'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sequence, and the correct answer. The output sequence is possibly truncated, as indicated by '[...]', and may contain tool calls. Your task is to determine if the given output sequence accurately answers the question based on the provided correct answer. Respond exactly and only with either [Correct] or [Incorrect].

---
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
---

Question: """{str(query)}"""

Output sequence: """{str(response)}"""

Correct answer: {str(ground_truth)}

Judgement:'''
    return prompt


async def eval_func_xverify_async(item, llm):
    """Async xVerify evaluation using the LLM.

    Expects `llm.inference` to be async and return a dict
    with a "response" key.
    """
    response = item["response"]
    response_len = len(response)
    if response_len > MAX_RESPONSE_LENGTH:
        prefix = "[...] "
        keep = MAX_RESPONSE_LENGTH - len(prefix)
        response = prefix + response[-keep:]
        logger.debug(
            f"Truncating response for evaluation. Removed {response_len - MAX_RESPONSE_LENGTH + 6} chars."
        )

    try:
        prompt = format_prompt(item["query"], response, item["gt"])
        llm_output = await llm.inference({"query": prompt})
        response = llm_output.get("response")
        valid_label = ["correct", "incorrect"]

        if isinstance(response, str):
            item_label = response.strip().lower()
            # Allow simple bracketed variants like "[correct]"
            normalized = item_label.strip("[]")
            if normalized in valid_label:
                if normalized == "correct":
                    return normalized, 1
                else:
                    return normalized, 0
            else:
                return f"Eval Error: {item_label}", None
        else:
            return "Eval Error: response is not a string", None
    except Exception as e:
        return f"Eval Error: {str(e)}", None
