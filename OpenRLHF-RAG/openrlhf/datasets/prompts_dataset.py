from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]
        sys_prompt = "You are skilled at solving problems through step-by-step reasoning, leveraging both your own knowledge and external search engine."
        user_prompt='''Given a complex multi-hop question, you need to reason step by step and **enclose the final answer within `<answer></answer>` tags**.
You have the ability to perform web searches. For uncertain knowledge, you **can utilize the external Search Engine to retrieve knowledge for solving questions**. You need to **provide the search query (only keywords) enclosed in the `<query></query>` tag**, then you will receive the relevant documents enclosed in the <tool_call></tool_call> tags.

The reasoning process should include detailed considerations such as analyzing questions, decomposing the questions, performing searching, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

During this process, you should use casual, genuine phrases such as: "Hmm...", "Wait, let me think about...", "Actually...", "Aha...", "Now that I look at it...", "This reminds me of...", "I wonder if...", "But then again...", "Let's see if...", "Alternatively...", "Let's summarize existing information...", etc., to make the reasoning process coherent, clear, and logically sound, effectively simulating human cognitive processes.

**Guidelines**:
- You should **show the reasoning process**, and **ONLY return the FINAL ANSWER within `<answer></answer>` tags**. For example: The Reasoning Process...<answer>The Final Answer</answer>.
- When you need to retrieve external knowledge, you should **provide the search query (only keywords) enclosed in the `<query></query>` tag**. For example: <query>Query consist of Keywords</query>.
- When done searching, continue your reasoning.

[Question]
{question}
'''
        user_prompt_new='''Given a complex multi-hop question, you need to reason step by step and **enclose the final answer within `<answer></answer>` tags**.
You have the ability to perform web searches. For uncertain knowledge, you **can utilize the external Search Engine to retrieve knowledge for solving questions**. You need to **provide the search query (only keywords) enclosed in the <|begin_of_query|><|end_of_query|> tags**, then you will receive the relevant documents enclosed in the <|begin_of_documents|><|end_of_documents|> tags.

The reasoning process should include detailed considerations such as analyzing questions, decomposing the questions, performing searching, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of current steps, refining any errors, and revisiting previous steps.

During this process, you should use casual, genuine phrases such as: "Hmm...", "Wait, let me think about...", "Actually...", "Aha...", "Now that I look at it...", "This reminds me of...", "I wonder if...", "But then again...", "Let's see if...", "Alternatively...", "Let's summarize existing information...", etc., to make the reasoning process coherent, clear, and logically sound, effectively simulating human cognitive processes.

**Guidelines**:
- You should **show the reasoning process**, and **ONLY return the FINAL ANSWER within `<answer></answer>` tags**. For example: The Reasoning Process...<answer>The Final Answer</answer>.
- When you need to retrieve external knowledge, you should **provide the search query (only keywords) enclosed in the `<|begin_of_query|><|end_of_query|>` tag**. For example: <|begin_of_query|>Query consist of Keywords<|end_of_query|>.
- When done searching, continue your reasoning.

[Question]
{question}
'''
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_new.format(question=question)},
        ]

        messages_chat_v1=[
            {"role": "system","content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""},
            {"role": "user", "content":question}
        ]

        messages_chat_v3=[
            {"role": "system","content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""},
            {"role": "user", "content":question}
        ]

        user_prompt_dpsk="""You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>".\n\n[Question]\n{question}"""
        messages_chat_v1_dpsk=[
            {"role": "user", "content":user_prompt_dpsk.format(question=question)}
        ]
        prompt = apply_chat_template(messages_chat_v3, tokenize=False, add_generation_prompt=True) + "<think>"

    else:
        base_prompt_v1 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""
        base_prompt_v2 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        base_prompt_v3 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

        question = data["question"]
        idx = data["idx"]
        prompt = base_prompt_v3.format(question=question)

    return str(idx) + "<|idx_prompt_split|>" + prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
        # print("len(self.prompts):",len(self.prompts))
        # print("self.prompts[0:5]:",self.prompts[0:5])
        # kill

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
