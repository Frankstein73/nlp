import gradio as gr
import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import tempfile
from pathlib import Path
import src.TAALED.process_fn as process_fn_taaled
import src.TAMMI.process_fn as process_fn_tammi

# --- 依赖检查与模型加载 ---

nlp = spacy.load("en_core_web_sm")


def figure_to_numpy(fig) -> np.ndarray:
    """
    将 Matplotlib Figure 对象渲染并转换为 NumPy 数组 (RGBA 格式)。

    Args:
        fig: 要转换的 Matplotlib Figure 对象。

    Returns:
        一个 NumPy 数组，形状为 (height, width, 4)，
        表示图像的 RGBA 像素数据，数据类型为 uint8。

    Raises:
        AttributeError: 如果 Figure 对象没有 'canvas' 属性 (不常见)。
        ValueError: 如果无法获取画布尺寸或缓冲区。
    """
    if not hasattr(fig, "canvas") or fig.canvas is None:
        raise AttributeError(
            "输入的 Figure 对象似乎没有 'canvas' 属性或 canvas 为 None。"
        )

    # 1. 强制 Matplotlib 绘制图形到画布上
    #    这是关键一步，确保缓冲区包含最新的图形渲染结果。
    try:
        fig.canvas.draw()
    except Exception as e:
        # 捕获可能的绘制错误
        raise RuntimeError(f"在尝试绘制 Figure 到画布时出错: {e}") from e

    # 2. 从画布获取宽度和高度
    try:
        width, height = fig.canvas.get_width_height()
    except Exception as e:
        raise ValueError(f"无法从画布获取宽度/高度: {e}") from e

    if width <= 0 or height <= 0:
        raise ValueError(f"从画布获取的宽度或高度无效: width={width}, height={height}")

    # 3. 从画布获取 RGBA 像素数据缓冲区
    try:
        # buffer_rgba() 返回一个 memoryview
        rgba_buffer = fig.canvas.buffer_rgba()
    except Exception as e:
        raise ValueError(f"无法从画布获取 RGBA 缓冲区: {e}") from e

    # 4. 将缓冲区数据转换为 NumPy 数组
    try:
        # 使用 np.frombuffer 将 memoryview (或 bytes) 高效转换为 NumPy 数组
        image_array = np.frombuffer(rgba_buffer, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"将缓冲区转换为 NumPy 数组时出错: {e}") from e

    # 5. 重塑数组为 (height, width, 4) 的形状
    expected_size = height * width * 4
    if image_array.size != expected_size:
        raise ValueError(
            f"缓冲区大小 ({image_array.size}) 与预期的画布尺寸 ({height}x{width}x4={expected_size}) 不匹配。"
        )

    try:
        image_array = image_array.reshape(height, width, 4)
    except ValueError as e:
        # 这个错误理论上不应发生，如果前面大小检查通过的话，但以防万一
        raise ValueError(f"无法将数组重塑为 ({height}, {width}, 4) 的形状: {e}") from e

    return image_array


def calculate_lexical_diversity(text):
    refined_lemma_dict = process_fn_taaled.tag_processor_spaCy(
        text,
        adj_word_list_path="src/TAALED/dep_files/adj_lem_list.txt",
        real_word_list_path="src/TAALED/dep_files/real_words.txt",
    )
    lemma_text_cw = refined_lemma_dict["content"]
    lemma_text_fw = refined_lemma_dict["function"]
    [lexical_density_tokens, lexical_density_types] = process_fn_taaled.lex_density(
        lemma_text_cw, lemma_text_fw
    )
    return lexical_density_tokens


def calculate_mci(text):
    metric = process_fn_tammi.analyze_text(
        text, "src/TAMMI/morpho_lex_df_w_log_w_prefsuf_no_head.csv"
    )
    return metric["inflectional MCI (10)"], metric["derivational MCI (10)"]


# --- 1. 标点符号替换函数 ---
def replace_punctuation(text):
    """将中文标点替换为英文标点"""
    replacements = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "、": ",",
        "…": "...",
        " ": " ",  # 保留英文空格
        "\u3000": " ",  # 全角空格替换为半角空格
        "\n": "\n",  # 保留换行符
        "\t": "\t",  # 保留制表符
    }
    # 替换每个标点符号
    for zh, en in replacements.items():
        text = text.replace(zh, en)
    return text


# --- 2. 文本分词函数 ---
def split_words(text):
    """使用 spaCy 进行分词，只保留字母组成的词"""
    doc = nlp(text)
    # 只保留纯字母构成的 token
    words = [token.text for token in doc if token.is_alpha]
    return words, sum(len(word) for word in words) / len(words)


# --- 3. 绘制词长分布图函数 ---
def plot_length_distribution(words, title):
    """绘制词长分布图并返回 matplotlib Figure 对象"""
    if not words:
        print(f"警告: 文件 '{title}' 没有找到有效单词进行绘图。")
        # 返回一个空白或带提示的图像
        fig = plt.figure(figsize=(10, 6))
        plt.title(f"{title}\n(No alphabetic words found)")
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        # plt.close(fig) # Close the figure object immediately after creation if empty? Maybe not needed.
        process_fig = figure_to_numpy(fig)
        plt.close(fig)
        return process_fig  # Return the figure object

    word_lengths = [len(word) for word in words]
    if not word_lengths:
        print(f"警告: 文件 '{title}' 的单词长度列表为空。")
        fig = plt.figure(figsize=(10, 6))
        plt.title(f"{title}\n(No valid word lengths)")
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        process_fig = figure_to_numpy(fig)
        plt.close(fig)
        return process_fig  # Return the figure object

    word_length_counts = Counter(word_lengths)
    # 按词长排序以获得更好的绘图效果
    x = sorted(list(word_length_counts.keys()))
    y = [word_length_counts[k] for k in x]

    # 创建一个新的 Figure 对象，避免全局状态干扰
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)  # Add subplot to the figure
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel("Word Length")
    ax.set_ylabel("Frequency")

    process_fig = figure_to_numpy(fig)
    plt.close(fig)
    return process_fig  # Return the figure object


# --- 4. 指标计算函数 (模拟 TAALED) ---
def calculate_metrics_simulated(essay, filename):
    """模拟计算 TTR 和 MATTR"""
    if not essay:
        return f"**{Path(filename).name}:**\n- 无法计算指标 (未找到文本)\n"

    refined_lemma_dict = process_fn_taaled.tag_processor_spaCy(
        essay,
        adj_word_list_path="src/TAALED/dep_files/adj_lem_list.txt",
        real_word_list_path="src/TAALED/dep_files/real_words.txt",
    )  # Process the essay using the functions in process_fn
    # Extract the lemma text and the content and function words
    lemma_text_aw = refined_lemma_dict["lemma"]
    # lemma_text_cw = refined_lemma_dict["content"]
    # lemma_text_fw = refined_lemma_dict["function"]
    simple_ttr = process_fn_taaled.simple_ttr(lemma_text_aw)
    mattr_aw_50 = process_fn_taaled.mattr(lemma_text_aw, 50)
    return simple_ttr, mattr_aw_50


# --- Gradio 核心处理函数 ---
def process_text_files(
    files,
    replace_punct,
    do_wordlength,
    do_plotting,
    do_metrics,
    do_lexical_diversity,
    do_mci,
):
    """处理上传的文件，根据选项执行流水线步骤"""
    if files is None:
        return "请先上传 .txt 文件。", [], []  # 返回与 outputs 对应的空值

    output_plots = []
    output_files_paths = []  # 用于存储处理后文件的路径
    metrics_data = []
    temp_dir = tempfile.mkdtemp()

    for file_obj in files:
        original_filename = Path(file_obj.name).name  # 获取原始文件名
        print(f"--- Processing file: {original_filename} ---")

        try:
            # 读取文件内容
            with open(file_obj.name, "r", encoding="utf-8-sig") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file {original_filename}: {e}")
            continue  # 处理下一个文件

        processed_text = text  # 初始化处理后的文本

        # --- 步骤 1: 标点符号替换 (可选) ---
        if replace_punct:
            print(f"  - Replacing punctuation for {original_filename}...")
            processed_text = replace_punctuation(processed_text)

        words = []
        words_avg_len = None
        # --- 步骤 2 & 3 & 4 依赖的分词 ---
        # 仅当需要绘图或计算指标时才进行分词
        if do_plotting or do_wordlength:
            print(f"  - Splitting words for {original_filename}...")
            words, words_avg_len = split_words(
                processed_text
            )  # 使用可能已替换标点的文本

        # --- 步骤 3: 绘制词长分布图 (可选) ---
        if do_plotting:
            print(f"  - Plotting word length distribution for {original_filename}...")
            plot_title = f"Word Length Distribution: {original_filename}"
            try:
                fig = plot_length_distribution(words, plot_title)

                output_plots.append(fig)
            except Exception as e:
                print(f"Error plotting for {original_filename}: {e}")

        # --- 步骤 4: 计算指标 (可选) ---
        simple_ttr, mattr_aw_50 = None, None
        if do_metrics:
            print(f"  - Calculating metrics for {original_filename}...")
            try:
                # 使用模拟函数；如果要用 TAALED，替换下面这行
                simple_ttr, mattr_aw_50 = calculate_metrics_simulated(
                    processed_text, original_filename
                )
            except ImportError as e:
                print(f"ImportError for TAALED: {e}. Using simulated metrics.")
            except Exception as e:
                print(f"Error calculating metrics for {original_filename}: {e}")

        lexical_diversity = None
        if do_lexical_diversity:
            print(f"  - Calculating lexical diversity for {original_filename}...")
            try:
                lexical_diversity = calculate_lexical_diversity(processed_text)
            except Exception as e:
                print(
                    f"Error calculating lexical diversity for {original_filename}: {e}"
                )

        inflectional_mci, derivational_mci = None, None
        if do_mci:
            print(f"  - Calculating MCI for {original_filename}...")
            try:
                inflectional_mci, derivational_mci = calculate_mci(processed_text)
            except Exception as e:
                print(f"Error calculating MCI for {original_filename}: {e}")

        # --- 保存处理后的文本以供下载 ---
        output_filename = f"processed_{original_filename}"
        output_path = os.path.join(temp_dir, output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            output_files_paths.append(output_path)
            print(f"  - Saved processed text to {output_path}")
        except Exception as e:
            print(f"Error writing processed file {output_filename}: {e}")

        metrics_data.append(
            [
                original_filename,
                words_avg_len,
                simple_ttr,
                mattr_aw_50,
                lexical_diversity,
                inflectional_mci,
                derivational_mci,
            ]
        )

    return metrics_data, output_plots, output_files_paths


# --- 构建 Gradio Blocks 界面 ---
with gr.Blocks(title="Workshop 1") as demo:
    gr.Markdown("# Workshop 1")
    gr.Markdown(
        "上传一个或多个 `.txt` 文件，选择要执行的处理步骤，然后点击“处理文件”按钮。"
    )

    with gr.Row():
        # 左侧：输入和控制选项
        with gr.Column(scale=1):
            input_files = gr.File(
                label="上传 TXT 文件",
                file_count="multiple",  # 允许多文件上传
                file_types=[".txt"],  # 限制文件类型
                type="filepath",  # 获取临时文件路径
            )
            gr.Markdown("### 可选处理步骤:")
            cb_replace_punct = gr.Checkbox(label="替换中文标点为英文标点", value=True)
            cb_do_wordlength = gr.Checkbox(label="计算词长", value=True)
            cb_do_plotting = gr.Checkbox(label="绘制词长分布图", value=True)
            cb_do_metrics = gr.Checkbox(label="计算文本指标 (TTR, MATTR)", value=True)
            cb_do_lexical_diversity = gr.Checkbox(label="计算词汇多样性", value=True)
            cb_do_mci = gr.Checkbox(label="计算词形变化指标", value=True)

            process_button = gr.Button("处理文件", variant="primary")

            download_output = gr.File(
                label="下载处理后的文件",
                file_count="multiple",  # 允许下载多个文件
            )

        # 右侧：输出结果
        with gr.Column(scale=2):
            plot_output = gr.Gallery(
                label="词长分布图",
                elem_id="plot-gallery",  # 可选，用于 CSS
                columns=3,  # 每行显示多少个图
                object_fit="contain",  # 图片适应方式
                height="auto",  # 高度自适应
            )
            gr.Markdown("### 处理结果:")
            # metrics_output = gr.Markdown(label="计算指标")
            metrics_output = gr.Dataframe(
                headers=[
                    "File Name",
                    "Avg Word Length",
                    "TTR",
                    "MATTR@50",
                    "Lexical Diversity",
                    "Inflectional MCI (10)",
                    "Derivational MCI (10)",
                ],
                col_count=(7, "fixed"),
            )

    # --- 连接按钮点击事件与处理函数 ---
    process_button.click(
        fn=process_text_files,
        inputs=[
            input_files,
            cb_replace_punct,
            cb_do_wordlength,
            cb_do_plotting,
            cb_do_metrics,
            cb_do_lexical_diversity,
            cb_do_mci,
        ],
        outputs=[metrics_output, plot_output, download_output],
        queue=True,
    )

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    # demo.launch(share=True) # 如果需要生成公开链接
    demo.queue()
    demo.launch()
