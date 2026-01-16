from loguru import logger

from .agentverse import AgentVerse_HumanEval, AgentVerse_Main, AgentVerse_MGSM
from .camel import CAMEL_Main
from .cot import CoT
from .dylan import DyLAN_HumanEval, DyLAN_Main, DyLAN_MATH, DyLAN_MMLU
from .llm_debate import LLM_Debate_Main
from .mad import MAD_Main
from .mas_base import MAS
from .mav import MAV_GPQA, MAV_MATH, MAV_MMLU, MAV_HumanEval, MAV_Main
from .self_consistency import SelfConsistency

# unsafe:
# from .autogen import AutoGen_Main
# from .evomac import EvoMAC_Main
# from .chatdev import ChatDev_SRDD
# from .macnet import MacNet_Main, MacNet_SRDD
# from .mapcoder import MapCoder_HumanEval, MapCoder_MBPP

method2class = {
    "vanilla": MAS,
    "cot": CoT,
    "agentverse_humaneval": AgentVerse_HumanEval,
    "agentverse_mgsm": AgentVerse_MGSM,
    "agentverse": AgentVerse_Main,
    "llm_debate": LLM_Debate_Main,
    "dylan_humaneval": DyLAN_HumanEval,
    "dylan_math": DyLAN_MATH,
    "dylan_mmlu": DyLAN_MMLU,
    "dylan": DyLAN_Main,
    "camel": CAMEL_Main,
    "mad": MAD_Main,
    "self_consistency": SelfConsistency,
    "mav": MAV_Main,
    "mav_main": MAV_Main,
    "mav_gpqa": MAV_GPQA,
    "mav_humaneval": MAV_HumanEval,
    "mav_math": MAV_MATH,
    "mav_mmlu": MAV_MMLU,
    # "autogen": AutoGen_Main,
    # "evomac": EvoMAC_Main,
    # "chatdev_srdd": ChatDev_SRDD,
    # "macnet": MacNet_Main,
    # "macnet_srdd": MacNet_SRDD,
    # "mapcoder_humaneval": MapCoder_HumanEval,
    # "mapcoder_mbpp": MapCoder_MBPP,
}


def get_method_class(method_name, dataset_name=None):
    # lowercase the method name
    method_name = method_name.lower()

    all_method_names = method2class.keys()
    matched_method_names = [
        sample_method_name
        for sample_method_name in all_method_names
        if method_name in sample_method_name
    ]

    if len(matched_method_names) > 0:
        if dataset_name is not None:
            # lowercase the dataset name
            dataset_name = dataset_name.lower()
            # check if there are method names that contain the dataset name
            matched_method_data_names = [
                sample_method_name
                for sample_method_name in matched_method_names
                if sample_method_name.split("_")[-1] in dataset_name
            ]
            if len(matched_method_data_names) > 0:
                method_name = matched_method_data_names[0]
                if len(matched_method_data_names) > 1:
                    logger.warning(
                        f"[WARNING] Found multiple methods matching {dataset_name}: {matched_method_data_names}. Using {method_name} instead."
                    )
    else:
        logger.error(
            f"[ERROR] No method found matching {method_name}. Please check the method name."
        )
        raise ValueError(
            f"[ERROR] No method found matching {method_name}. Please check the method name."
        )

    return method2class[method_name]
