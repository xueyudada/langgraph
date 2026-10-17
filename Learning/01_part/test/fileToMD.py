import os
import re
import base64
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

def convert_doc_to_docx(doc_path):
    """在 Windows 环境下将 .doc 转换为 .docx (需要安装 Word 及 pywin32)"""
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
        doc.SaveAs2(abs_docx_path, FileFormat=16)  # 16 代表 docx 格式
        doc.Close()
    finally:
        word.Quit()
    return abs_docx_path

def encode_image(image_path):
    """读取本地图片并转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_image_description(image_path, llm):
    """调用多模态大模型来描述图片"""
    try:
        base64_image = encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "你是一个图像识别专家，请为这张图片生成极其详细的中文描述。如果是图表，请列出其中的核心数据和趋势；如果是架构图/流程图，请描述其逻辑关系和节点步骤；如果是截屏或插图，请提取出其中的关键文字和主要内容。不要包含任何多余的前言和后缀，直接给出描述内容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"识别图片 {image_path} 失败: {e}")
        return None

def replace_images_in_markdown(md_path, llm):
    """扫描 Markdown 中的本地图片引用，调用大模型识别后添加描述文本"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配 Markdown 图片格式，跳过 http 网络链接
    pattern = re.compile(r'(!\[(.*?)\]\(((?!http).+?\.(?:png|jpg|jpeg|gif|webp|bmp))\))')
    matches = pattern.findall(content)
    
    if not matches:
        print("未在 Markdown 中发现可识别的本地图片。")
        return

    print(f"检测到 {len(matches)} 处本地图片引用，开始调用多模态大模型进行内容识别并生成描述...")
    
    md_dir = Path(md_path).parent
    replaced_count = 0
    processed_images = {}

    for full_match, alt_text, img_rel_path in matches:
        # 根据 markdown 中的相对路径拼接出图片的绝对路径
        img_abs_path = md_dir / img_rel_path
        
        if not img_abs_path.exists():
            print(f"图片未在磁盘上找到，跳过: {img_abs_path}")
            continue
            
        if img_rel_path in processed_images:
            description = processed_images[img_rel_path]
        else:
            print(f"正在识别: {img_rel_path} ...")
            description = get_image_description(img_abs_path, llm)
            if description:
                processed_images[img_rel_path] = description
            
        if description:
            # 格式：保留原图，并在其下方插入多模态大模型生成的描述（利于 RAG 进行文本切片与向量检索）
            replacement = f"{full_match}\n\n【图片描述：{description}】"
            content = content.replace(full_match, replacement)
            replaced_count += 1

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"成功为 {replaced_count} 处图片添加了多模态文本描述！")

def file_to_markdown(file_path, enable_vision=False):
    # 如果是 .doc 格式，先转换为 .docx
    if file_path.lower().endswith(".doc"):
        print(f"检测到旧版 .doc 格式，正在转换为 .docx...")
        try:
            file_path = convert_doc_to_docx(file_path)
        except Exception as e:
            print(f"转换 .doc 失败: {e}。建议手动另存为 .docx 后重试。")
            return

    file_path_obj = Path(file_path)
    
    # 配置 PipelineOptions，开启图片生成与 OCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # 启用 OCR
    pipeline_options.generate_picture_images = True  # 生成图片，以便导出保存
    pipeline_options.generate_page_images = True  # 可选：如果需要导出整页图片可以开启
    
    # 初始化 Converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"正在将 {file_path} 转换为 Markdown...")
    result = converter.convert(file_path_obj)
    
    # 设定输出路径为当前工作目录
    md_path = Path.cwd() / file_path_obj.name
    md_path = md_path.with_suffix(".md")
    
    # 将结果保存为 Markdown 文件，同时将图片保存到本地文件夹
    print("正在保存 Markdown 文件及图片...")
    result.document.save_as_markdown(
        filename=md_path,
        image_mode=ImageRefMode.REFERENCED
    )
    print(f"转换成功！Markdown 已保存至: {md_path}")
    print(f"图片提取成功！若有图片，已被保存至同级目录下对应的图片文件夹。")

    # 如果启用了多模态视觉描述替换
    if enable_vision:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_URL")
        model_name = os.getenv("MODEL_NAME")
        
        if not api_key:
            print("错误：未找到 OPENAI_API_KEY 环境变量，无法启动多模态替换。")
            return
            
        print(f"正在初始化 Vision 大模型 ({model_name})...")
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.1
        )
        replace_images_in_markdown(md_path, llm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将任意文档转换为 Markdown 格式（支持图片提取、OCR 与 Vision 多模态图片描述）")
    parser.add_argument("file_path", nargs="?", help="要转换的文件路径")
    parser.add_argument("--vision", action="store_true", help="启用多模态大模型识别图片并替换为文本描述")
    args = parser.parse_args()

    file_path = args.file_path
    enable_vision = args.vision

    # 若命令行未指定路径，则进入交互式输入
    if not file_path:
        user_input = input("请输入要转换的文件路径 (直接回车将寻找当前目录下的默认 test.pdf): ").strip()
        if user_input:
            file_path = user_input

    # 询问是否开启多模态替换
    if not enable_vision:
        user_vision = input("是否启用多模态大模型识别图片并添加描述？(y/n，默认n): ").strip().lower()
        if user_vision == 'y':
            enable_vision = True

    # 默认路径检测
    if not file_path:
        script_dir = Path(__file__).parent
        test_pdf = script_dir / "test.pdf"
        
        if test_pdf.exists():
            file_path = str(test_pdf)
        else:
            print("错误：未指定文件路径，且当前目录下未找到默认的 test.pdf 文件。")
            sys.exit(1)

    file_to_markdown(file_path, enable_vision=enable_vision)
