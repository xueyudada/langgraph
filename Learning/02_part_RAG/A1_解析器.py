import os
import re
import sys
import base64
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from hierarchical.postprocessor import ResultPostprocessor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

"""
Docling 布局感知解析与多模态 VLM 图像清洗实现

本示例演示生产环境中 RAG Ingestion (文档解析与清洗) 的高级用法：
1. 采用 Docling 提取文档的物理布局（大纲、表格还原、图像坐标）。
2. 将输入文件 `text.pdf` 解析并保存为 `text_pdf.md`（原文件名+后缀名命名规则）。
3. 自动扫描文档中的图像。若有图片，调用 VLM 进行多模态视觉描述：
   - 提取图片的详细描述，并将其以 `【图片描述：...】` 追加到 Markdown 文本中以增强检索。
   - 让大模型为图片取名（如 `sales_chart`），并将图片文件重命名为 `text_pdf_{大模型的取名}.png`，同时自动同步修改 Markdown 中的引用链接。
"""

# 加载当前目录下的 .env 配置文件
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


def parse_document_to_markdown(file_path: str, enable_vision: bool = False):
    """
    使用 Docling 加载并解析文档，保存为 Markdown 和提取出的图片，最后视配置执行 VLM 图像语义清洗与重命名。
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

    # 1. 配置 Docling 渲染引擎选项（支持图像提取和 OCR）
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
    
    # 注入后处理器，修复内存中 Document 对象的标题层级关系
    print("正在注入后处理器修复标题层级关系...")
    postprocessor = ResultPostprocessor(result)
    postprocessor.process()
    
    # 2. 确定生成的 Markdown 文件的命名规则 (原文件名_后缀名.md)
    # 例如：text.pdf -> text_pdf.md
    base_name = file_path_obj.stem
    ext_name = file_path_obj.suffix.replace(".", "").lower()
    prefix_name = f"{base_name}_{ext_name}"
    
    # 输出文件路径定在当前工作目录下
    md_filename = f"{prefix_name}.md"
    md_path = Path.cwd() / md_filename
    
    # 3. 将解析出的文档和图片保存到本地磁盘
    # 使用 REFERENCED 模式会将提取到的图片保存到 {prefix_name}-assets 目录下并在 md 中引用
    print("正在写入本地 Markdown 及图片集...")
    result.document.save_as_markdown(
        filename=md_path,
        image_mode=ImageRefMode.REFERENCED
    )
    print(f"Markdown 成功保存至: {md_path}")
    
    # 4. 如果启用多模态图片识别
    if enable_vision:
        # 优先读取 Image 相关的变量，如果不存在则使用标准的 OPENAI_ 变量作为兜底
        api_key = os.getenv("Image_API_KEY")
        base_url = os.getenv("Image_BASE_URL")
        model_name = os.getenv("Image_MODEL_NAME")
        
        if not api_key:
            print("警告: 未检测到 OPENAI_API_KEY 或 Image_API_KEY，跳过 VLM 图片识别。")
            return
            
        print(f"初始化 VLM 多模态大模型 ({model_name})...")
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.1
        )
        
        process_images_and_rename(md_path, prefix_name, llm)


def main():
    parser = argparse.ArgumentParser(description="A1_解析器: 布局感知解析器(Docling) + VLM 语义增强")
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
        # 检查当前目录下是否有 test.pdf
        test_pdf = Path.cwd() / "test.pdf"
        if test_pdf.exists():
            file_path = str(test_pdf)
        else:
            # 检查同级 test 目录中是否有 test.pdf (符合原示例目录结构)
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
