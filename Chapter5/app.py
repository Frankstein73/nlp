import gradio as gr
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import DataLoader
from preprocess_data import read_files, CustomDataset
from model_roberta import RoBERTaClass
import os
import warnings
from sklearn.preprocessing import LabelEncoder
import tempfile
import glob

warnings.filterwarnings("ignore")  # ignore warnings

CUDA_DEVICE = f"cuda:{os.environ.get('CUDA_DEVICE', '0')}"


MODEL_NAME_OR_PATH = (
    "./roberta_large"  # the path to the model configuration file and tokenizer
)
CKPT_PATH = "./checkpoint.pt"  # the path to the model checkpoint file

checkpoint = torch.load(CKPT_PATH, map_location="cpu")  # load the model checkpoint
model_checkpoint = {
    k.replace("module.", "", 1): v for k, v in checkpoint["model_state_dict"].items()
}  # convert the model checkpoint to the correct format, remove the "module." prefix if it exists
label_encoder = LabelEncoder()
label_encoder.classes_ = checkpoint["classes"]  # load the label encoder
model = RoBERTaClass(
    MODEL_NAME_OR_PATH, len(label_encoder.classes_)
)  # initialize the model
model.load_state_dict(model_checkpoint)  # load the model checkpoint
model.eval()  # set the model to evaluation mode
model.to(CUDA_DEVICE)  # move the model to the correct device
tokenizer = RobertaTokenizerFast.from_pretrained(
    MODEL_NAME_OR_PATH
)  # initialize the tokenizer


@torch.no_grad()
def process_text_files(input_files):
    temp_dir = tempfile.mkdtemp()
    text_list, _, _, file_path_list, original_text_list = read_files(
        file_list=input_files, is_test=True
    )  # read files from the test data directory
    for text, file_path, original_text in zip(
        text_list, file_path_list, original_text_list
    ):
        print(f"Processing {file_path}")  # print the file path
        test_set = CustomDataset(
            text, [], tokenizer, is_test=True
        )  # initialize the test dataset
        test_loader = DataLoader(
            test_set, batch_size=5, shuffle=False
        )  # initialize the test loader
        all_preds = []  # initialize the list to store the predicted labels

        for i, data in enumerate(test_loader):
            ids = data["ids"].to(
                CUDA_DEVICE, dtype=torch.long
            )  # move the ids to the correct device
            mask = data["mask"].to(
                CUDA_DEVICE, dtype=torch.long
            )  # move the mask to the correct device
            special_tokens_mask = data["special_tokens_mask"].to(
                CUDA_DEVICE, dtype=torch.long
            )  # move the special tokens mask to the correct device
            outputs = model(ids, mask, special_tokens_mask)  # get the model outputs
            preds = torch.argmax(outputs, dim=1)  # get the predicted labels
            all_preds.extend(
                preds.cpu().detach().numpy().tolist()
            )  # store the predicted labels

        all_preds_str = [
            label_encoder.classes_[pred]  # type: ignore
            for pred in all_preds
        ]  # convert the predicted labels to strings

        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(temp_dir, base_name.replace(".txt", "_pred.txt"))

        with open(new_file_path, "w", encoding="utf-8") as f:  # open the output file
            for i in range(len(original_text)):
                if (
                    all_preds_str[i] == "[none]"
                ):  # if the predicted label is [none], write the original text
                    f.write(original_text[i])
                elif (
                    original_text[i][-1] == "\n"
                ):  # if the original text ends with a newline character, write the original text without the newline character and the predicted label
                    f.write(original_text[i][:-1] + all_preds_str[i] + "\n")
                else:  # otherwise, write the original text and the predicted label
                    f.write(original_text[i] + all_preds_str[i] + " ")

    return [file_path for file_path in glob.glob(temp_dir + "/*.txt")]


with gr.Blocks(title="Workshop 3") as demo:
    gr.Markdown("# Workshop 3")
    gr.Markdown("上传一个或多个 `.txt` 文件，点击“处理文件”按钮。")

    input_files = gr.File(
        label="上传 TXT 文件",
        file_count="multiple",  # 允许多文件上传
        file_types=[".txt"],  # 限制文件类型
        type="filepath",  # 获取临时文件路径
    )

    process_button = gr.Button("处理文件", variant="primary")

    download_output = gr.File(
        label="下载处理后的文件",
        file_count="multiple",  # 允许下载多个文件
    )

    # --- 连接按钮点击事件与处理函数 ---
    process_button.click(
        fn=process_text_files,
        inputs=[
            input_files,
        ],
        outputs=[download_output],
        queue=True,
    )

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
