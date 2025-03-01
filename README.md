
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

Large reasoning models (LRMs), such as OpnAI-o1 and Deepseek-R1, have demonstrated the significant impact of reinforcement learning in enhancing the long-step reasoning capabilities of models, thereby greatly improving their reasoning performance. Despite these advantages, when faced with knowledge-intensive problems, especially multi-hop questions and time-sensitive issues, these models may lack the necessary knowledge. Therefore, it is great important to enable LLMs to invoke web search and obtain external information during the reasoning process. 

We propose **Reason-Searcher**, utilizing a *two-stage outcome-supervision reinforcement learning* approach to enable the model to learn to invoke web search during the reasoning process: first allowing the model to learn how to invoke web search, and then teaching it how to effectively use that search engine. This method does not require any instruction fine-tuning for cold start, and at the same time, it is compatible with existing Base LLMs or Chat LLMs. We will open-source the training code, inference code, model checkpoints, and the detailed technical report.

- Model:
    - Qwen-2.5-7B-Base-RAG-RL: https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL
    - Llama-3.1-8B-Instruct-RAG-RL: https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL
- Train-data:  https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki
- Technical report: coming soon…
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
