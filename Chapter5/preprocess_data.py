import glob
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

RANDOM_STATE = 16
MAX_LENGTH = 512

# 定义标签映射
LABEL_MAP = {
    "[claim]": "[claim]",
    "[cliam]": "[claim]",
    "[data]": "[data]",
    "[dara]": "[data]",
    "[ddata]": "[data]",
    "[counterclaim]": "[counterclaim]",
    "[rebuttal]": "[rebuttal]",
}


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, is_test=False):
        self.tokenizer = tokenizer
        self.labels = labels
        self.texts = texts
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        inputs = self.tokenizer(
            text,
            None,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        special_tokens_mask = [
            1 if id == self.tokenizer.sep_token_id else 0 for id in ids
        ]  # 根据实际ID调整

        # Ignore tail [SEP]
        special_tokens_mask[-1] = 0
        ret = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_tokens_mask, dtype=torch.long),
        }
        if not self.is_test:
            ret["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return ret


def read_files(
    dir_path: str | None = None,
    file_list: list[str] | None = None,
    run_name: str = "default",
    is_test: bool = False,
):
    docs = []
    all_labels = []
    assert (dir_path is None) != (file_list is None), (
        "dir_path and file_list cannot both be None"
    )
    if dir_path is not None:
        print("Reading files from ", dir_path)
        for file_path in glob.glob(dir_path + "/*.txt"):
            texts, labels, doc_contexts = read_file(file_path, is_test)
            all_labels.extend(labels)
            docs.append((texts, labels, doc_contexts, file_path))
    else:
        print("Reading files from ", file_list)
        for file_path in file_list:
            texts, labels, doc_contexts = read_file(file_path, is_test)
            all_labels.extend(labels)
            docs.append((texts, labels, doc_contexts, file_path))

    print(docs[0])
    # 对标签进行编码
    label_encoder = LabelEncoder()
    file_paths = [doc[3] for doc in docs]
    original_text = [doc[2] for doc in docs]
    if not is_test:
        print(pd.Series(all_labels).value_counts())

        label_encoder.fit(all_labels)
        print(label_encoder.classes_, len(label_encoder.classes_))

        text_test, label_test = process_docs(docs, label_encoder)

    else:
        text_test, label_test = [doc[0] for doc in docs], []

    return text_test, label_test, label_encoder, file_paths, original_text


def write_to_dir(dir_path, suffix, docs, run_name):
    for texts, labels, doc_contexts, file_path in docs:
        new_file_path = file_path.replace(
            dir_path, "ckpt/" + run_name + "/" + dir_path + suffix
        )
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        with open(new_file_path, "w", encoding="utf-8") as f:
            for i in range(len(doc_contexts)):
                label = labels[i]
                # 写入文本和标签
                if doc_contexts[i][-1] == "\n":
                    f.write(doc_contexts[i][:-1] + " " + label + "\n")
                else:
                    f.write(doc_contexts[i] + " " + label + "\n")


def read_file(file_path, is_test=False):
    texts = []
    labels = []
    doc_contexts = []

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f.readlines():
            # 将全角字符转换为半角字符
            line = strQ2B(line)

            # 使用正则表达式提取句子和标签
            # print(line)
            regex_results = (
                re.findall(r"(.*?[\.\?!]+)\s*(\[[\w-]+\])", line)
                if not is_test
                else re.findall(r"(.*?[\.\?!]+)", line)
            )

            # 对提取的句子和标签进行处理
            for res in regex_results:
                # print(res)
                if not is_test:
                    unit_text = res[0].strip()  # 提取句子
                    label = res[1].strip()  # 提取标签
                    # 检查标签是否在映射表中
                    if label not in LABEL_MAP:
                        continue
                    # 更新为映射后的标签
                    label = LABEL_MAP[label]
                    # 将句子和标签加入列表
                    labels.append(label)
                    doc_contexts.append(unit_text)
                else:
                    unit_text = res.strip()
                    doc_contexts.append(unit_text)
    # print(doc_contexts)
    # 构建包含上下文的文本列表
    for i in range(len(doc_contexts)):
        # 构建 before_context 和 after_context
        before_context = "".join(doc_contexts[:i])
        after_context = "".join(doc_contexts[i + 1 :])
        overall_length = len(before_context) + len(doc_contexts[i]) + len(after_context)

        # 如果整体长度超过 2400，截断上下文
        if overall_length > 2400:
            truncate_length = max(len(before_context) - (overall_length - 2400), 1000)
            before_context = before_context[-truncate_length:]

        # 构造最终的文本格式: before_context [SEP] sentence [SEP] after_context
        combined_text = (
            before_context + " [SEP] " + doc_contexts[i] + " [SEP] " + after_context
        )
        texts.append(combined_text)

    return texts, labels, doc_contexts


def strQ2B(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (
                inside_code >= 65281 and inside_code <= 65374
            ):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def process_docs(docs, label_encoder):
    """处理文档，返回文本和标签"""
    texts = []
    labels = []
    for text_list, label_list, doc_contexts, file_path in docs:
        texts.extend(text_list)
        encoded_labels = label_encoder.transform(label_list)
        labels.extend(encoded_labels)
        assert len(encoded_labels) == len(doc_contexts)
    return texts, labels
