import os
import re
from typing import List, Dict, Any, Set, Tuple

from methods.mas_base import MAS
from .prompt_main import *


# Define the NEWMAS class which inherits from MAS and implements the inference method
class AgentVerse_Main(MAS):
    def __init__(self, general_config, method_config_name=None):
        method_config_name = (
            "config_main" if method_config_name is None else method_config_name
        )
        super().__init__(general_config, method_config_name)

        self.max_turn = self.method_config["max_turn"]
        self.cnt_agents = self.method_config["cnt_agents"]
        self.max_criticizing_rounds = self.method_config["max_criticizing_rounds"]

        self.dimensions: List[str] = ["Score", "Response"]
        self.advice = "No advice yet."
        self.history = []

    async def inference(self, sample) -> dict[str, str]:
        query = sample["query"]
        solution: str = ""

        for _ in range(self.max_turn):
            # Assign roles to agents
            role_descriptions = await self.assign_roles(query)

            # Collaborate to solve the query
            solution = await self.group_vertical_solver_first(query, role_descriptions)

            # Get evaluation and feedback
            score, feedback = await self.evaluate(query, role_descriptions, solution)

            if score == 1:
                break
            else:
                self.advice = feedback
        return {"response": solution}

    async def assign_roles(self, query: str):
        # Fetch prompts from config.yaml (assumed to be loaded earlier)
        prepend_prompt = (
            ROLE_ASSIGNER_PREPEND_PROMPT.replace("${query}", query)
            .replace("${cnt_agents}", str(self.cnt_agents))
            .replace("${advice}", self.advice)
        )
        append_prompt = ROLE_ASSIGNER_APPEND_PROMPT.replace(
            "${cnt_agents}", str(self.cnt_agents)
        )

        # Call LLM (role assigner logical agent) to get role assignments
        assigner_messages = self.construct_messages(prepend_prompt, [], append_prompt)
        role_assignment_response = await self.call_llm_for_agent_async(
            agent_id="agentverse_assigner", messages=assigner_messages
        )
        # Extract role descriptions using regex
        role_descriptions = self.extract_role_descriptions(role_assignment_response)
        return role_descriptions

    def extract_role_descriptions(self, response: str):
        """
        Extracts the role descriptions from the model's response using regex.
        Assumes the response is formatted like:
        1. an electrical engineer specified in the field of xxx.
        2. an economist who is good at xxx.
        ...
        """
        role_pattern = (
            r"\d+\.\s*([^.]+)"  # extract the content between the number and the period
        )

        role_descriptions = re.findall(role_pattern, response)

        if len(role_descriptions) == self.cnt_agents:
            # print("role_descriptions:")
            # print(role_descriptions)
            return role_descriptions
        else:
            raise ValueError(
                f"wrong cnt_agent, expect {self.cnt_agents} agents while we find {len(role_descriptions)} role_descriptions."
            )

    async def group_vertical_solver_first(
        self, query: str, role_descriptions: List[str]
    ):
        max_history_solver = 5
        max_history_critic = 3
        previous_plan = "No solution yet."
        nonempty_reviews = []
        history_solver = []
        history_critic = []

        if self.advice != "No advice yet.":
            self.history.append(
                {"role": "assistant", "content": f"[Evaluator]: {self.advice}"}
            )
            history_solver = (
                self.history[-max_history_solver:]
                if len(self.history) > max_history_solver
                else self.history
            )

        # Step 1: Solver generates a solution
        solver_prepend_prompt = SOLVER_PREPEND_PROMPT.replace("${query}", query)
        solver_append_prompt = SOLVER_APPEND_PROMPT.replace(
            "${role_description}", role_descriptions[0]
        )
        solver_message = self.construct_messages(
            solver_prepend_prompt, history_solver, solver_append_prompt
        )
        solver_response = await self.call_llm_for_agent_async(
            agent_id="agentverse_solver", messages=solver_message
        )
        self.history.append(
            {
                "role": "assistant",
                "content": f"[{role_descriptions[0]}]: {solver_response}",
            }
        )
        history_critic = (
            self.history[-max_history_critic:]
            if len(self.history) > max_history_critic
            else self.history
        )
        previous_plan = solver_response

        cnt_critic_agent = self.cnt_agents - 1

        for _ in range(self.max_criticizing_rounds):
            reviews = []
            for j in range(cnt_critic_agent):
                critic_prepend_prompt = CRITIC_PREPEND_PROMPT.replace(
                    "${query}", query
                ).replace("${role_description}", role_descriptions[j + 1])
                critic_append_prompt = CRITIC_APPEND_PROMPT
                critic_message = self.construct_messages(
                    critic_prepend_prompt, history_critic, critic_append_prompt
                )
                critic_response = await self.call_llm_for_agent_async(
                    agent_id="agentverse_critic", messages=critic_message
                )
                if "[Agree]" not in critic_response:
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": f"[{role_descriptions[j + 1]}]: {self.parse_critic(critic_response)}",
                        }
                    )
                    history_solver = (
                        self.history[-max_history_solver:]
                        if len(self.history) > max_history_solver
                        else self.history
                    )
                reviews.append(critic_response)

            for review in reviews:
                if "[Agree]" not in review:
                    nonempty_reviews.append(review)
            if not nonempty_reviews:
                break

            solver_message = self.construct_messages(
                solver_prepend_prompt, history_solver, solver_append_prompt
            )
            solver_response = await self.call_llm_for_agent_async(
                agent_id="agentverse_solver", messages=solver_message
            )
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"[{role_descriptions[0]}]: {solver_response}",
                }
            )
            history_critic = (
                self.history[-max_history_critic:]
                if len(self.history) > max_history_critic
                else self.history
            )
            previous_plan = solver_response

        return previous_plan

    def parse_critic(self, output) -> str:
        output = re.sub(r"\n+", "\n", output.strip())
        if "[Agree]" in output:
            return ""
        else:
            return output

    async def evaluate(self, query: str, role_descriptions: List[str], Plan):
        evaluator_prepend_prompt = (
            EVALUATOR_PREPEND_PROMPT.replace("${query}", query)
            .replace("${all_role_description}", "\n".join(role_descriptions))
            .replace("${solution}", Plan)
        )
        evaluator_append_prompt = EVALUATOR_APPEND_PROMPT
        evaluator_message = self.construct_messages(
            evaluator_prepend_prompt, [], evaluator_append_prompt
        )
        evaluator_response = await self.call_llm_for_agent_async(
            agent_id="agentverse_evaluator", messages=evaluator_message
        )
        return self.parse_evaluator(evaluator_response)

    def parse_evaluator(self, output) -> Tuple[int, str]:
        # Support both "Correctness:" and "Score:" style labels and fall back
        # to the first 0/1 digit if needed.
        correctness_match = re.search(r"(Correctness|Score):\s*(\d)", output)
        if correctness_match:
            correctness = int(correctness_match.group(2))
        else:
            fallback_match = re.search(r"\b([01])\b", output)
            if fallback_match:
                correctness = int(fallback_match.group(1))
            else:
                raise ValueError("Correctness not found in the output text.")

        advice_match = re.search(r"Response:\s*(.+)", output, re.DOTALL)
        if advice_match:
            advice = advice_match.group(1).strip()
            clean_advice = re.sub(r"\n+", "\n", advice.strip())
        else:
            # If the model deviates from the expected format, treat the
            # whole output as advice instead of failing hard.
            clean_advice = re.sub(r"\n+", "\n", output.strip())

        return correctness, clean_advice

    def construct_messages(
        self, prepend_prompt: str, history: List[Dict], append_prompt: str
    ):
        messages = []
        if prepend_prompt:
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt:
            messages.append({"role": "user", "content": append_prompt})
        return messages
