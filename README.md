
<h1 align="center"> XX: xxxxx</a></h1>


<div align="center"> 
<a href="https://github.com/SsmallSong/RLRAG/edit/main//LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
<a href="https://github.com/SsmallSong/RLRAG/edit/main//LICENSE"><img src="https://img.shields.io/badge/Model_License-MIT-blue" alt="license"></a>
<a href="[https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3](https://github.com/SsmallSong/RLRAG)"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
 
</div>


<!-- <div align="center">
    <span style="display:inline-block; margin-right: 10px;">
        <a href="https://paperswithcode.com/sota/mathematical-reasoning-on-aime24?p=search-o1-agentic-search-enhanced-large">
            <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/search-o1-agentic-search-enhanced-large/mathematical-reasoning-on-aime24" alt="AIME24 Badge">
        </a>
    </span>
    <span style="display:inline-block; margin-right: 10px;">
        <a href="https://paperswithcode.com/sota/mathematical-reasoning-on-amc23?p=search-o1-agentic-search-enhanced-large">
            <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/search-o1-agentic-search-enhanced-large/mathematical-reasoning-on-amc23" alt="AMC23 Badge">
        </a>
    </span>
  <span style="display:inline-block; margin-right: 10px;">
        <a href="https://paperswithcode.com/sota/on-gpqa?p=search-o1-agentic-search-enhanced-large">
            <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/search-o1-agentic-search-enhanced-large/on-gpqa" alt="GPQA Badge">
        </a>
    </span>
</div> -->



<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **02/28/2025**: The Qwen-2.5-7B-RAGRL and Llama-3.1-8B-RAGRL have released, you can find them here : 

## 💡 Overview



大型推理模型（LRMs），如OpenAI-o1, Deepseek-R1，展示了强化学习在提升模型的长步骤推理能力的显著作用，进而大幅度提升模型推理能力。尽管这些模型具有优势，但是面对知识密集型的问题，尤其是多跳问题和时间敏感性问题，可能缺少需要的知识。而RAG（retrieval a？ generation）通过进行检索来获取外部知识帮助模型进行推理回答。
我们将强化学习和RAG结合起来，使用二阶段结果监督RL，先让模型学习调用搜索引擎，再让模型学习如何调用搜索引擎。无需复杂的prompt设计和流程设计，让模型学会自己使用RAG，平衡内外部知识

我们在Qwen-2.5-7B-base和Llama3.1-8B-instruct进行了训练，并将训练代码，推理代码，模型checkpoint，详细的技术报告全部开源。

Large reasoning models (LRMs), such as OpenAI-o1 and Deepseek-R1, have demonstrated the significant impact of reinforcement learning in enhancing the long-step reasoning capabilities of models, thereby greatly improving their reasoning performance. Despite these advantages, when faced with knowledge-intensive problems, especially multi-hop questions and time-sensitive issues, these models may lack the necessary knowledge. Retrieval-Augmented Generation (RAG) helps models by retrieving external knowledge to assist in reasoning responses.

We combine reinforcement learning and RAG using a two-stage result-supervised RL approach: first allowing the model to learn how to invoke a search engine, and then teaching it how to effectively use that search engine. This method eliminates the need for complex prompt and process designs, enabling the model to learn to use RAG independently while balancing internal and external knowledge.

We have trained our approach on Qwen-2.5-7B-base and Llama3.1-8B-instruct, and we have open-sourced the training code, inference code, model checkpoints, and a detailed technical report.

## ✨ Method


## 🔗 Model Downloads 


## 📄 Benchmarks
### Settings
### Results

## 🏃 Quick Start



## 📄 Citation



## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [3151273556@qq.com](3151273556@qq.com).
