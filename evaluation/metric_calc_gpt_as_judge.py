from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
import json
from tqdm import tqdm
import google.generativeai as genai
import multiprocessing
from time import sleep
import requests
import json
from collections import defaultdict
import random
import json
import requests
import time
# from bs4 import BeautifulSoup
# import wikipediaapi
from urllib.parse import unquote
from urllib.parse import urlparse

# ================== CONFIG ==================
# Set your Google API key (export GOOGLE_API_KEY in env or paste here)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBZb6PfzwULVP9rjwZzW7kwdfDIrO8tGOU")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "models/gemini-1.5-pro-latest"  # or "gemini-1.5-pro"

# ================== GEMINI CALL ==================
def generate(prompt, model_name=MODEL_NAME):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip()


def process_one_sample(obj):
    prompt = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer.
Respond with True if the prediction is correct and False otherwise.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
    '''

    question = obj["question"]
    reference_ans = obj["answer"]
    if reference_ans is False:
        reference_ans = "no"
    if reference_ans is True:
        reference_ans = "yes"

    proposed_ans = obj.get("pred_ans", "N/A")

    gemini_input = prompt.format(
        question=question,
        reference=reference_ans,
        prediction=proposed_ans
    )

    try:
        model_output = generate(gemini_input, MODEL_NAME)
        obj["gemini_output"] = model_output
    except Exception as e:
        obj["gemini_output"] = f"ERROR: {str(e)}"

    try:
        print("=="*70)
        print("Q:", question)
        print("Pred:", proposed_ans)
        print("Gold:", reference_ans)
        print("Judge:", obj["gemini_output"])
        print("=="*70)
    except:
        pass

    return obj


if __name__ == '__main__':
    input_files=[
    "/content/Capstone_UOA/data/eval_set/bamboogle-Capstone_UOAQwen-2.5-7B-base-RAG-RL_base_temp0.0_typev0.jsonl",
    ]
    # base_path="/home/songhuatong/RAG-Long-Cot/sys_data/data/a_seed_sft_new_new_new/origianl_data"
    # files= os.listdir(base_path)
    # input_files = [base_path+"/"+file for file in files]

    for input_file in input_files:
        output_file = input_file.replace(".jsonl","_judge.jsonl")
        chunk_size=400

        with open(input_file,"r") as fin:
            all_demons = fin.readlines()
            all_demons = [json.loads(s) for s in all_demons]

        print("All Data Num:",len(all_demons))
        chunk_num = len(all_demons) // chunk_size
        if len(all_demons) % chunk_size != 0:
            chunk_num += 1

        for chunk_i in range(chunk_num):
            print("Epoch:" ,chunk_i ,"/",chunk_num)
            all_demons_subset = all_demons[chunk_i*chunk_size : (chunk_i+1)*chunk_size]
            print(len(all_demons_subset))
            with multiprocessing.Pool(processes=400) as pool:
                results = list(tqdm(pool.imap(process_one_sample, all_demons_subset), total=len(all_demons_subset)))

            with open(output_file, 'a') as fout:
                for res in results:
                    if res is not None:
                        fout.write(json.dumps(res)+'\n')


