from methods.mas_base import MAS


class CoT(MAS):
    def __init__(self, general_config, method_config_name=None):
        # CoT is a simple method that doesn't use configurations
        # method_config_name is accepted for compatibility with inference script
        super().__init__(general_config, method_config_name)

    async def inference(self, sample):
        prompt = sample["query"] + "\n\nLet's think step by step."

        response = await self.call_llm(prompt=prompt)

        return {"response": response}
