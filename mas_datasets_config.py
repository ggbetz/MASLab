SUPPORTED_DATASETS = {
    "MATH",
    "GSM8K",
    "AQUA-RAT",
    "MedMCQA",
    "MedQA",
    "MMLU",
    "MMLU-Pro",
    "GSM-Hard",
    "SciBench",
    "AIME-2024",
    "AIME-2025",
    "APEX-SHORTLIST",
}


def dataset_is_supported(name: str) -> bool:
    if name in SUPPORTED_DATASETS:
        return True
    if name.startswith("GPQA"):
        return True
    return False
