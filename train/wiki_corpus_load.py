
from flask import Flask, request, jsonify
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import os
import time
import sys
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    # 从请求中获取查询向量
    data = request.json

    queries = data["queries"]

    k = data.get("k", 3)

    # s = time.time()
    query_embeddings = model.encode_queries(queries)

    all_answers = []
    D, I = index.search(query_embeddings, k=k)  # 假设返回前3个结果
    for idx in I:
        answers_for_query = [corpus[i] for i in idx[:k]] # 找出该query对应的k个答案
        all_answers.append(answers_for_query)  # 将该query的答案列表存储

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    data_type = sys.argv[1]
    port = sys.argv[2]

    model = FlagModel(
        "/opt/aps/workdir/model/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=False,
    )

    print("模型已加载完毕")

    # 加载语料库
    if data_type == 'hotpotqa':
        file_path = "/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv"
    elif data_type =="all":
        file_path ="/opt/aps/workdir/sht-RAG_RL/train/wiki_server/all_wiki/nq/output.tsv"
    elif data_type =="with_2wiki":
        file_path ="/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-20171001-pages-meta-current-withlinks-abstracts_add_2wiki.tsv"
    elif data_type =="kilt":
        file_path ="/opt/aps/workdir/model/kilt_100/wiki_kilt_100_really.tsv"

    else:
        kill
        file_path = "/opt/aps/workdir/input/data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv"
    corpus = load_corpus(file_path)

    print(f"语料库已加载完毕-{len(corpus)}")

    # 加载建好的索引
    if data_type == 'hotpotqa':
        index_path = "/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-abs-index_w_title-bge-large-en-v1.5.bin"
    elif data_type=="all":
        index_path = "/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-all-index_w_title-bge-large-en-v1.5.bin"
    elif data_type =="with_2wiki":
        index_path ="/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-abs-index_w_title_add_2wiki.bin"
    elif data_type =="kilt":
        index_path ="/opt/aps/workdir/model/kilt_100/enwiki_kilt_all.bin"

    else:
        kill
        index_path = "/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki-abs-index_w_title-bge-large-en-v1.5.bin"
    index = faiss.read_index(index_path)

    print("索引已经建好")
# /opt/aps/workdir/input/data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv

    app.run(host="0.0.0.0", port=port, debug=False)  # 在本地监听端口5003
    print("可以开始查询")

