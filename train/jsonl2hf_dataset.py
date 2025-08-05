from datasets import load_dataset

# 加载你的 .jsonl 文件为数据集
dataset = load_dataset("json", data_files="xx.jsonl")
print(dataset)
# 打印数据集的一部分查看内容


# 将数据集保存到原目录
dataset["train"].save_to_disk("xx") #folder name
