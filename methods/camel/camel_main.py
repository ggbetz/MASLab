import re

from dataclasses import dataclass

from ..mas_base import MAS
from .prompt_main import SystemPromptGenerator


@dataclass
class _ChoiceMessage:
    message: object


@dataclass
class _CompletionLike:
    choices: list[_ChoiceMessage]


class CAMEL_Main(MAS):
    def __init__(self, model_config, method_config_name=None):
        method_config_name = (
            "config_main" if method_config_name is None else method_config_name
        )
        super().__init__(model_config, method_config_name)

        self.chat_turn_limit = self.method_config["chat_turn_limit"]
        self.assistant_role = self.method_config["assistant_role"]
        self.user_role = self.method_config["user_role"]
        self.system_prompt_generator = SystemPromptGenerator()
        self.with_critic = self.method_config["with_critic"]
        self.option_num = self.method_config["option_num"]
        self.critic_role = self.method_config["critic_role"]

    async def _generate_options(
        self, agent_id: str, messages, option_num: int
    ) -> _CompletionLike:
        """Generate multiple independent options using MAS async LLM.

        Returns an object with a .choices list, where each element has a
        .message.content attribute, mimicking the OpenAI API structure CAMEL
        expects.
        """

        choices: list[_ChoiceMessage] = []
        for _ in range(option_num):
            content = await self.call_llm_for_agent_async(
                agent_id=agent_id, messages=messages
            )
            msg = type("Msg", (), {"content": content})
            choices.append(_ChoiceMessage(message=msg))

        return _CompletionLike(choices=choices)

    async def inference(self, sample):
        query = sample["query"]

        # Task specification phase: use user agent
        (
            _,
            _,
            _,
            task_specify_sys_msg,
            task_specify_prompt,
            _,
        ) = self.system_prompt_generator.generate(
            self.assistant_role, self.user_role, query
        )

        task_specify_messages = [
            {"role": "system", "content": task_specify_sys_msg},
            {"role": "user", "content": task_specify_prompt},
        ]

        specified_task_msg = await self.call_llm_for_agent_async(
            agent_id="camel_user", messages=task_specify_messages
        )

        response = f"Original idea prompt: {query}\n\n"
        response += f"Specified task prompt: {specified_task_msg}\n\n"

        if self.with_critic:
            # System prompts for assistant, user, critic
            (
                assistant_sys_msg,
                user_sys_msg,
                user_prompt,
                _,
                _,
                critic_sys_msg,
            ) = self.system_prompt_generator.generate(
                self.assistant_role,
                self.user_role,
                specified_task_msg,
                critic_role=self.critic_role,
            )
            self.user_messages = [
                {"role": "system", "content": user_sys_msg},
                {"role": "user", "content": user_prompt},
            ]
            self.assistant_messages = [{"role": "system", "content": assistant_sys_msg}]
            self.critic_messages = [{"role": "system", "content": critic_sys_msg}]

            for _ in range(self.chat_turn_limit):
                # User multiple-choice proposals
                user_completion = await self._generate_options(
                    agent_id="camel_user",
                    messages=self.user_messages,
                    option_num=self.option_num,
                )

                user_response = self.form_user_response(
                    user_completion, self.option_num
                )
                response += f"User Message: \n{user_response}\n\n"

                # Critic selects among user proposals
                self.critic_messages.append({"role": "user", "content": user_response})
                critic_response = await self.call_llm_for_agent_async(
                    agent_id="camel_critic", messages=self.critic_messages
                )
                response += f"Critic Message: \n{critic_response}\n\n"
                self.critic_messages.append(
                    {"role": "assistant", "content": critic_response}
                )
                selected_option = self.find_option(critic_response)
                if selected_option is None:
                    selected_option = 1
                selected_user_response = getattr(
                    user_completion.choices[selected_option - 1].message, "content", ""
                )

                # Assistant multiple-choice based on selected user response
                self.user_messages.append(
                    {"role": "assistant", "content": selected_user_response}
                )
                self.assistant_messages.append(
                    {"role": "user", "content": selected_user_response}
                )
                assistant_completion = await self._generate_options(
                    agent_id="camel_assistant",
                    messages=self.assistant_messages,
                    option_num=self.option_num,
                )
                assistant_response = self.form_assistant_response(
                    assistant_completion, self.option_num
                )
                response += f"Assistant Message: \n{assistant_response}\n\n"

                # Critic over assistant proposals
                self.critic_messages.append(
                    {"role": "user", "content": assistant_response}
                )
                selected_option = self.find_option(critic_response)
                if selected_option is None:
                    selected_option = 1
                selected_assistant_response = getattr(
                    assistant_completion.choices[selected_option - 1].message,
                    "content",
                    "",
                )
                self.assistant_messages.append(
                    {"role": "assistant", "content": selected_assistant_response}
                )
                self.user_messages.append(
                    {"role": "user", "content": selected_assistant_response}
                )

                if "CAMEL_TASK_DONE" in selected_user_response:
                    break
        else:
            # No critic: simple user-assistant conversation
            (
                assistant_sys_msg,
                user_sys_msg,
                user_prompt,
                _,
                _,
                _,
            ) = self.system_prompt_generator.generate(
                self.assistant_role, self.user_role, specified_task_msg
            )
            self.user_messages = [
                {"role": "system", "content": user_sys_msg},
                {"role": "user", "content": user_prompt},
            ]
            self.assistant_messages = [{"role": "system", "content": assistant_sys_msg}]

            for _ in range(self.chat_turn_limit):
                user_response = await self.call_llm_for_agent_async(
                    agent_id="camel_user", messages=self.user_messages
                )
                self.user_messages.append(
                    {"role": "assistant", "content": user_response}
                )
                self.assistant_messages.append(
                    {"role": "user", "content": user_response}
                )
                response += f"User Message: \n{user_response}\n\n"

                assistant_response = await self.call_llm_for_agent_async(
                    agent_id="camel_assistant", messages=self.assistant_messages
                )
                self.assistant_messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
                self.user_messages.append(
                    {"role": "user", "content": assistant_response}
                )
                response += f"Assistant Message: \n{assistant_response}\n\n"

                if user_response is None:
                    break
                if "CAMEL_TASK_DONE" in user_response:
                    response += "Assistant Message: \nGreat! Let me know if you have any other tasks or questions."
                    break

        return {"response": response}

    def form_user_response(self, completion, option_num):
        # Form the user response with the multiple choice options to a standard format
        response = f"""> Proposals from {self.user_role} (RoleType.USER). Please choose an option:\n"""
        for i in range(option_num):
            response += (
                f"""Option {i + 1}:\n{completion.choices[i].message.content}\n"""
            )
        response += f"""Please first enter your choice ([1-{option_num}]) and then your explanation and comparison: """
        return response

    def form_assistant_response(self, completion, option_num):
        # Form the assistant response with the multiple choice options to a standard format
        response = f"""> Proposals from {self.assistant_role} (RoleType.ASSISTANT). Please choose an option:\n"""
        for i in range(option_num):
            response += (
                f"""Option {i + 1}:\n{completion.choices[i].message.content}\n"""
            )
        response += f"""Please first enter your choice ([1-{option_num}]) and then your explanation and comparison: """
        return response

    def find_option(self, string):
        # Find the first integer number found in the given string. It means the choice of the critic agent
        match = re.search(r"\d+", string)
        if match:
            return int(match.group())
        else:
            return None
