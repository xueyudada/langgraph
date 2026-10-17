import os
import re
import sys
import base64
import shutil
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

"""
Docling 布局感知解析 + LLM 智能标题层级重构 + VLM 图像清洗实现 (测试版)

本示例解决 Docling 解析 PDF 时所有标题均为二级的局限。
采用双大模型流水线：
1. 提取所有 H2 标题和上下文片段，利用通用/对话大模型 (如 DeepSeek) 重构文档大纲层级结构。
2. (选填) 自动扫描文档中的图像，利用多模态大模型 (如 GLM-4V) 重构并生成图片详细中文描述并重命名。
"""

# 加载 .env 配置文件
load_dotenv()


def convert_doc_to_docx(doc_path: str) -> str:
    """在 Windows 环境下将旧版 .doc 转换为 .docx"""
    try:
        import win32com.client
    except ImportError:
        raise ImportError("解析 .doc 文件需要安装 pywin32，请运行: pip install pywin32")
    
    abs_doc_path = os.path.abspath(doc_path)
    abs_docx_path = abs_doc_path + "x"
    
    if os.path.exists(abs_docx_path):
        return abs_docx_path
        
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(abs_doc_path)
        doc.SaveAs2(abs_docx_path, FileFormat=16)  # 16 代表 docx
        doc.Close()
    finally:
        word.Quit()
    return abs_docx_path


def encode_image_base64(image_path: Path) -> str:
    """读取本地图片文件并编码为 Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_vlm(image_path: Path, llm: ChatOpenAI) -> tuple[str, str]:
    """
    调用多模态大模型对图像进行双重分析：
    1. 为图片命名的英文/拼音简短名字 (用于重命名文件)
    2. 图片的详细中文语义描述 (用于 RAG 语义增强)
    """
    try:
        base64_image = encode_image_base64(image_path)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "你是一个图像识别专家。请为这张图片完成两个任务：\n"
                        "1. 起一个简短的英文或拼音名字，用于给该图片文件重命名（只能用小写字母、数字和下划线，不超过3个单词，例如 'sales_trend'，'system_architecture'）。\n"
                        "2. 针对该图片的极其详细的中文描述（如果是图表请提取数据和趋势，如果是架构图/流程图请描述步骤逻辑）。不要包含前言和后缀。\n\n"
                        "请严格按以下格式输出：\n"
                        "NAME: <图片名字>\n"
                        "DESCRIPTION: <中文详细描述>"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
        response = llm.invoke([message])
        res_text = response.content.strip()
        
        # 解析输出
        name_match = re.search(r"NAME:\s*(.+)", res_text)
        desc_match = re.search(r"DESCRIPTION:\s*(.+)", res_text, re.DOTALL)
        
        vlm_name = name_match.group(1).strip() if name_match else "unnamed"
        # 移除非法字符，防范文件名报错
        vlm_name = re.sub(r"[^a-zA-Z0-9_]", "", vlm_name.replace(" ", "_").lower())
        
        description = desc_match.group(1).strip() if desc_match else res_text
        
        return vlm_name, description
    except Exception as e:
        print(f"VLM 解析图片 {image_path.name} 失败: {e}")
        return "unnamed_img", f"无法解析的图片内容: {e}"


def process_images_and_rename(md_path: Path, prefix_name: str, llm: ChatOpenAI):
    """
    扫描 Markdown 中的本地图片引用：
    1. 调用 VLM 获取大模型取名与文本描述。
    2. 将磁盘上的图片重命名为: {prefix_name}_{大模型的取名}.png
    3. 更新 Markdown 中的图片路径引用，并在下方追加 VLM 生成的图片中文描述。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配 Markdown 图片语法，如 ![alt](path)，排除 http 网络链接
    pattern = re.compile(r'(!\[(.*?)\]\(((?!http).+?\.(?:png|jpg|jpeg|gif|webp|bmp))\))')
    matches = pattern.findall(content)
    
    if not matches:
        print("未在转换后的 Markdown 中发现本地图片引用。")
        return
        
    print(f"发现 {len(matches)} 处本地图片，启动 VLM 多模态识别与重命名流...")
    
    md_dir = md_path.parent
    processed_images = {} # 避免同一个图片被重复请求

    for full_match, alt_text, img_rel_path in matches:
        # 获取图片的绝对路径
        img_abs_path = (md_dir / img_rel_path).resolve()
        
        if not img_abs_path.exists():
            print(f"警告: 未在磁盘上找到图片 {img_abs_path}，跳过处理。")
            continue
            
        # 若已处理过该相对路径的图片，直接复用结果
        if img_rel_path in processed_images:
            vlm_name, description, new_rel_path = processed_images[img_rel_path]
        else:
            print(f"正在分析图片: {img_rel_path} ...")
            vlm_name, description = analyze_image_with_vlm(img_abs_path, llm)
            
            # 拼接新的图片文件名: {prefix_name}_{大模型的取名}.png
            # 保持原图的后缀名
            suffix = img_abs_path.suffix
            new_filename = f"{prefix_name}_{vlm_name}{suffix}"
            
            # 新的绝对路径和相对路径
            img_folder = img_abs_path.parent
            new_abs_path = img_folder / new_filename
            new_rel_path = Path(img_rel_path).parent / new_filename
            
            # 在磁盘上重命名文件
            try:
                if img_abs_path != new_abs_path:
                    # 如果新文件已存在，先删除，防止 Windows 重命名冲突
                    if new_abs_path.exists():
                        os.remove(new_abs_path)
                    shutil.move(str(img_abs_path), str(new_abs_path))
                    print(f"重命名成功: {img_abs_path.name} -> {new_filename}")
            except Exception as e:
                print(f"重命名图片失败: {e}")
                new_rel_path = img_rel_path  # 失败则保留原相对路径
                
            processed_images[img_rel_path] = (vlm_name, description, str(new_rel_path).replace("\\", "/"))
            
        # 替换 Markdown 中的链接及注入描述
        new_img_markdown = f"![{alt_text}]({new_rel_path})"
        replacement = f"{new_img_markdown}\n\n【图片描述：{description}】"
        content = content.replace(full_match, replacement)

    # 写回更新后的 Markdown 文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"多模态图像重命名与描述注入处理完成！")


def fix_markdown_headings_with_llm(md_path: Path, llm: ChatOpenAI):
    """
    通过提取 Markdown 文件中的所有扁平化标题与首段正文片段，
    利用通用大模型 (LLM) 推理出正确的大纲级别并写回。
    该方法既不会修改/篡改正文，也无需发送整篇长文本给大模型，极低成本且安全。
    """
    if not md_path.exists():
        print(f"警告: 未找到 Markdown 文件 {md_path}，跳过标题层级修正。")
        return

    print("正在提取 Markdown 中的标题大纲...")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    headings = []
    
    # 扫描获取所有以 ## 开头的扁平标题
    for i, line in enumerate(lines):
        if line.startswith("## "):
            headings.append({"index": i, "text": line[3:].strip()})
            
    if not headings:
        print("未在 Markdown 中发现任何 H2 (##) 标题，跳过层级重写。")
        return

    # 构造请求上下文，提取标题下方最多 8 行的非空首句，用于 LLM 辨识子集归属
    headings_info = []
    for idx, h in enumerate(headings):
        line_idx = h["index"]
        context_lines = []
        for j in range(line_idx + 1, min(line_idx + 8, len(lines))):
            curr_line = lines[j].strip()
            # 如果遇到下一个标题，立即停止收集上下文
            if curr_line.startswith("## ") or curr_line.startswith("# ") or curr_line.startswith("### "):
                break
            if curr_line:
                context_lines.append(curr_line)
        context = " | ".join(context_lines)[:120]  # 控制字符数
        headings_info.append({
            "id": idx,
            "text": h["text"],
            "snippet": context
        })

    print(f"共提取出 {len(headings)} 个扁平标题。正在调用大模型推理层级结构...")
    
    prompt = (
        "你是一个文档结构大纲解析与纠错专家。下面是从一篇 PDF 文档导出的扁平化标题列表，其中每个标题均以“## ”开头，并附带了它下方的开头正文片段作为上下文。\n"
        "请仔细识别并执行以下任务：\n"
        "1. **层级判定**：根据每个标题的文字含义、正文片段线索、以及标题顺序之间的递进包含关系，推断并纠正每个标题合理的 Markdown 层级级别（level，取值范围 1-6，1代表最顶级标题，2为次级标题，3为更次级，以此类推）。\n"
        "2. **非标题识别（关键）**：如果某个被误标记为“##”的内容其实**不是标题**（例如它实际上是普通正文段落、列表项、表格单元格内容或噪声），请将它的 `level` 设为 `0`。\n\n"
        "【特别要求】\n"
        "1. 必须保留原标题的自然顺序，分析其逻辑嵌套深度。\n"
        "2. 不要漏掉或跳过任何标题，必须输出每个标题 id 的层级。\n"
        "3. 直接以 JSON 数组格式返回结果，禁止添加任何 Markdown 包裹符 (如 ```json) 或前后废话。\n\n"
        "大纲列表数据：\n"
        f"{json.dumps(headings_info, ensure_ascii=False, indent=2)}\n\n"
        "请严格按以下 JSON 数组格式返回：\n"
        "[\n"
        "  {\"id\": 0, \"level\": 1},\n"
        "  {\"id\": 1, \"level\": 2},\n"
        "  {\"id\": 2, \"level\": 0}\n"
        "]"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        res_text = response.content.strip()
        
        # 兼容处理大模型输出的 markdown 代码块包裹
        if res_text.startswith("```"):
            res_text = re.sub(r"^```(?:json)?\n", "", res_text)
            res_text = re.sub(r"\n```$", "", res_text)
        
        levels_map = json.loads(res_text)
        id_to_level = {item["id"]: item["level"] for item in levels_map}
        
        # 将算好的 # 数量更新到原 lines 中
        for idx, h in enumerate(headings):
            level = id_to_level.get(idx, 2) # 兜底默认为二级
            if level == 0:
                # 还原为普通文本（去除 ## 前缀）
                lines[h["index"]] = h["text"]
                print(f"   - 识别非标题并还原为正文: \"{h['text']}\"")
            else:
                level = max(1, min(6, level))   # 限制层级在 1-6 级内
                lines[h["index"]] = "#" * level + " " + h["text"]
            
        updated_content = "\n".join(lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print("🎉 大模型标题层级修复及非标题清洗成功，Markdown 文件已成功修正！")
        
    except Exception as e:
        print(f"⚠️ 大模型修正标题层级失败: {e}，保留原版 Markdown。")


def parse_document_to_markdown(file_path: str, enable_vision: bool = False):
    """
    使用 Docling 加载并解析文档，保存为 Markdown 和提取出的图片，
    接着使用通用大模型修复标题层级，最后视配置执行 VLM 图像语义清洗与重命名。
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"未找到输入文件: {file_path}")

    # 如果是旧版 .doc，转为 .docx 后处理
    if file_path_obj.suffix.lower() == ".doc":
        print("检测到旧版 .doc，转换为 .docx 中...")
        try:
            docx_path = convert_doc_to_docx(str(file_path_obj))
            file_path_obj = Path(docx_path)
        except Exception as e:
            print(f"转换失败: {e}，请手动另存为 .docx 后重新运行。")
            return

    # 1. 配置 Docling 渲染引擎选项
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True                  # 开启 OCR 支持扫描件
    pipeline_options.generate_picture_images = True # 提取并生成图片
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"开始使用 Docling 解析文档: {file_path_obj.name} ...")
    result = converter.convert(file_path_obj)
    
    # 2. 确定生成的 Markdown 文件的命名规则
    base_name = file_path_obj.stem
    ext_name = file_path_obj.suffix.replace(".", "").lower()
    prefix_name = f"{base_name}_{ext_name}"
    
    md_filename = f"{prefix_name}.md"
    md_path = Path.cwd() / md_filename
    
    # 3. 将解析出的文档和图片保存到本地磁盘
    print("正在写入本地 Markdown 及图片集...")
    result.document.save_as_markdown(
        filename=md_path,
        image_mode=ImageRefMode.REFERENCED
    )
    print(f"Markdown 成功保存至: {md_path}")
    
    # 4. 初始化通用大模型 (从 .env 读取标准的 OPENAI_ 变量)，并调用层级修复逻辑
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    if api_key:
        print(f"初始化通用大模型 ({model_name}) 准备修复大纲层级...")
        text_llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.1
        )
        fix_markdown_headings_with_llm(md_path, text_llm)
    else:
        print("警告: 未检测到 OPENAI_API_KEY，跳过大模型标题层级修复。")
    
    # 5. 如果启用多模态图片识别
    if enable_vision:
        img_api_key = os.getenv("Image_API_KEY")
        img_base_url = os.getenv("Image_BASE_URL")
        img_model_name = os.getenv("Image_MODEL_NAME")
        
        if not img_api_key:
            print("警告: 未检测到 Image_API_KEY，跳过 VLM 图片识别与重命名。")
            return
            
        print(f"初始化 VLM 多模态大模型 ({img_model_name})...")
        vlm_llm = ChatOpenAI(
            api_key=img_api_key,
            base_url=img_base_url,
            model=img_model_name,
            temperature=0.1
        )
        
        process_images_and_rename(md_path, prefix_name, vlm_llm)


def main():
    parser = argparse.ArgumentParser(description="A1_解析器_test: 布局感知解析器(Docling) + LLM 标题层级修复")
    parser.add_argument("file_path", nargs="?", help="要解析的文档路径")
    parser.add_argument("--vision", action="store_true", help="是否启用 VLM 多模态图像识别与自动重命名")
    args = parser.parse_args()

    file_path = args.file_path
    enable_vision = args.vision

    # 交互式设定输入文件
    if not file_path:
        user_input = input("请输入要解析的文件绝对或相对路径 (直接回车寻找内置 test.pdf): ").strip()
        if user_input:
            file_path = user_input.strip('"')

    # 询问是否启动 VLM
    if not enable_vision:
        user_vision = input("是否启用 VLM 识别图片并重命名？(y/n，默认 n): ").strip().lower()
        if user_vision == 'y':
            enable_vision = True

    # 默认兜底文件检测
    if not file_path:
        test_pdf = Path.cwd() / "test.pdf"
        if test_pdf.exists():
            file_path = str(test_pdf)
        else:
            test_pdf_alt = Path(__file__).parent / "test" / "test.pdf"
            if test_pdf_alt.exists():
                file_path = str(test_pdf_alt)
            else:
                print("错误: 未指定输入文件，且未在当前工作空间中找到 test.pdf 默认文件。")
                sys.exit(1)

    try:
        parse_document_to_markdown(file_path, enable_vision=enable_vision)
    except Exception as e:
        print(f"\n❌ 解析处理出错: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
