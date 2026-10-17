import os
import re
import sys
import json
import hashlib
import unicodedata
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document

# 尝试载入大模型库（用于生产级语义纠错和指代消解）
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN_OPENAI = True
except ImportError:
    HAS_LANGCHAIN_OPENAI = False

# 尝试载入 BeautifulSoup (用于网页 HTML 剥离)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class RAGDocumentCleaner:
    """
    RAG 数据管道 2: 生产级文档清洗器 (RAGDocumentCleaner)
    
    职责：处于【加载 (Loader)】之后，【切分 (Splitter)】之前。
    解决痛点：
      1. 去噪 (Noise Removal)：清除多余格式、页眉页脚页码噪音，防污染向量库。
      2. 标准化 (Normalization)：Unicode 全半角标准化，OCR 识别错字硬替换及智能纠错。
      3. 语义增强 (Semantic Enrichment)：大模型指代消解（解决 Chunk 切碎后丢失上下文主语的问题）、
         无效内容垃圾块过滤、全文档 MD5 内容级去重。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        min_char_length: int = 10
    ):
        self.min_char_length = min_char_length
        self.noise_patterns: List[str] = []
        self.ocr_correction_map: Dict[str, str] = {}
        self.coreference_map: Dict[str, str] = {}
        self.blacklist_phrases: List[str] = []

        # 加载配置（优先外部 JSON 配置文件，无则使用内置标准规则）
        self._load_configurations(config_path)

    def _load_configurations(self, config_path: Optional[str]):
        """从外部 JSON 动态加载清洗规则，实现配置与代码的解耦"""
        default_config = {
            "noise_patterns": [
                r"(?i)page\s*\d+\s*(?:of\s*\d+)?",                     # Page 1, Page 2 of 10
                r"第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?",             # 第 1 页, 第 1/10 页
                r"版权所有\s*©\s*[\u4e00-\u9fa5\w\.\-]+(?:\s*公司)?",   # 版权所有 © XXX 公司
                r"(?i)copyright\s*©\s*[\w\.\-]+",                     # Copyright © XXX
                r"Confidential|内部机密|机密文件|仅供内部参考"
            ],
            "ocr_correction_map": {
                "数掘": "数据",
                "键字": "关键字",
                "算怯": "算法",
                "系銃": "系统",
                "模裂": "模型",
                "向畺": "向量",
                "图型": "图形",
                "打字": "输入"
            },
            "coreference_map": {
                "该系统": "LangGraph 状态机系统",
                "该公司": "Google DeepMind 团队",
                "该框架": "LangGraph 框架"
            },
            "blacklist_phrases": [
                "点击查看详情",
                "谢谢阅读",
                "扫码关注",
                "广告推广",
                "此处省略",
                "您的支持是我们最大的动力"
            ]
        }

        config = default_config
        if config_path:
            p = Path(config_path)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    print(f"ℹ️ 已成功从外部配置文件 {p.name} 载入清洗规则。")
                except Exception as e:
                    print(f"⚠️ 读取配置文件失败: {e}，将回退使用内置规则。")
            else:
                print(f"⚠️ 找不到配置文件: {p}，将回退使用内置规则。")

        self.noise_patterns = config.get("noise_patterns", default_config["noise_patterns"])
        self.ocr_correction_map = config.get("ocr_correction_map", default_config["ocr_correction_map"])
        self.coreference_map = config.get("coreference_map", default_config["coreference_map"])
        self.blacklist_phrases = config.get("blacklist_phrases", default_config["blacklist_phrases"])

    # ==========================================
    # 模块一: 结构级清洗 (去噪 Noise Removal)
    # ==========================================

    def strip_html_and_nav(self, text: str) -> str:
        """网页抓取去噪：剥离 HTML 标签及无用的侧边栏、导航栏"""
        if HAS_BS4:
            soup = BeautifulSoup(text, "html.parser")
            # 去除不可视和辅助导航元素
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            return soup.get_text()
        else:
            # 备用正则脱落标签
            text = re.sub(r"<script[^>]*?>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*?>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", "", text)
            return text

    def remove_headers_footers(self, text: str) -> str:
        """去除页眉页脚、机密印记、版权与页码"""
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """连续换行/空格归一化，保留自然段落"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 3个以上连续空行归纳为2个（保留Markdown的分段空行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 连续制表符和空格压缩为单个空格
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # ==========================================
    # 模块二: 文本级清洗 (标准化 Normalization)
    # ==========================================

    def unicode_normalize(self, text: str) -> str:
        """使用 Unicode NFKC 归一化，将全角字母数字等统一转为半角"""
        return unicodedata.normalize("NFKC", text)

    def correct_ocr_errors(self, text: str) -> str:
        """基于本地常用错词映射词典进行快速纠错"""
        for err, corr in self.ocr_correction_map.items():
            text = text.replace(err, corr)
        return text

    # ==========================================
    # 模块三: 语义级清洗 (内容增强与过滤)
    # ==========================================

    def resolve_coreference_by_rules(self, text: str) -> str:
        """基于简单字典映射替换代词"""
        for pronoun, target in self.coreference_map.items():
            text = text.replace(pronoun, target)
        return text

    def clean_via_llm(self, text: str, llm: Any) -> str:
        """
        [生产级核心] 调用大模型进行无死角的智能拼写纠错与语义指代消解。
        相较于硬编码规则，LLM 能够根据文章的前后语义进行更准确的主语还原和OCR乱码修正。
        """
        system_prompt = (
            "你是一个专业的 RAG 数据清洗助手。请对输入的文本进行精细化语义清洗，主要任务如下：\n"
            "1. 【拼写与 OCR 纠错】：修正明显的拼写错误和 OCR 乱码（例如把“数掘”修正为“数据”，“系銃”修正为“系统”）。\n"
            "2. 【指代消解】：结合语境，如果文本中存在指代不明的代词（如“该公司”、“这套系统”、“它”），在保证语义准确的前提下，将其替换为最可能指代的具体名称，使切片后的分块能够独立被理解。\n"
            "注意：你只负责进行这两项语义修正，绝对不要缩写或大幅删减原文内容，保持原文的其他段落和标点结构不变。直接输出清洗修正后的纯文本，不要包含任何前言、解释、注释或 Markdown 代码块。"
        )
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"待清洗文本：\n{text}")
            ]
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ 大模型清洗调用异常，将降级为本地硬编码规则。异常信息: {e}", file=sys.stderr)
            return ""

    def is_valid_content(self, text: str) -> bool:
        """过滤字数少、全是标点、或包含黑名单词的垃圾块"""
        clean_text = text.strip()
        if len(clean_text) < self.min_char_length:
            return False
        # 必须包含至少一个汉字或英文字符/数字
        if not re.search(r"[\w\u4e00-\u9fa5]", clean_text):
            return False
        # 匹配垃圾过滤黑名单
        for phrase in self.blacklist_phrases:
            if phrase in clean_text:
                return False
        return True

    def enrich_metadata(self, doc: Document) -> Document:
        """元数据增强 (计算 MD5 指纹、计算文本长度)"""
        metadata = doc.metadata.copy()
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        metadata["content_hash"] = content_hash
        metadata["content_length"] = len(doc.page_content)
        return Document(page_content=doc.page_content, metadata=metadata)

    # ==========================================
    # 核心集成流水线 (Pipeline)
    # ==========================================

    def clean_single_text(self, text: str, llm: Optional[Any] = None) -> str:
        """对纯文本进行流水线清洗"""
        # 1. 网页标签/导航栏剥离
        text = self.strip_html_and_nav(text)
        # 2. 页眉页脚去噪
        text = self.remove_headers_footers(text)
        # 3. Unicode 标准化
        text = self.unicode_normalize(text)
        
        # 4. 纠错与指代消解 (优先大模型，无则降级本地字典规则)
        if llm:
            llm_cleaned = self.clean_via_llm(text, llm)
            if llm_cleaned:
                text = llm_cleaned
            else:
                text = self.correct_ocr_errors(text)
                text = self.resolve_coreference_by_rules(text)
        else:
            text = self.correct_ocr_errors(text)
            text = self.resolve_coreference_by_rules(text)

        # 5. 空格多余格式压缩
        text = self.normalize_whitespace(text)
        return text

    def clean_document(self, doc: Document, llm: Optional[Any] = None) -> Optional[Document]:
        """清洗单个 Document 对象。不合原则被过滤时返回 None"""
        cleaned_text = self.clean_single_text(doc.page_content, llm)
        if not self.is_valid_content(cleaned_text):
            return None
        cleaned_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
        return self.enrich_metadata(cleaned_doc)

    def clean_and_deduplicate(self, docs: List[Document], llm: Optional[Any] = None) -> List[Document]:
        """批量清洗 Document 列表并全批次去重"""
        cleaned_docs: List[Document] = []
        seen_hashes: Set[str] = set()

        for doc in docs:
            cleaned = self.clean_document(doc, llm)
            if cleaned is None:
                continue

            doc_hash = cleaned.metadata.get("content_hash", "")
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                cleaned_docs.append(cleaned)

        return cleaned_docs


# =====================================================================
# 文件 I/O 读写与命令行入口
# =====================================================================
def process_file_or_dir(
    input_path: Path,
    cleaner: RAGDocumentCleaner,
    llm: Optional[Any] = None
) -> Tuple[int, int]:
    """读取实际的物理文件进行清洗，并在同级目录下输出结果"""
    files_to_process: List[Path] = []

    # 1. 递归扫描指定路径下的文件
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # 主要针对 Markdown (.md) 和 Text (.txt) 文件
        files_to_process.extend(input_path.glob("**/*.md"))
        files_to_process.extend(input_path.glob("**/*.txt"))

    if not files_to_process:
        print("⚠️ 未发现可处理的文件 (支持 .md, .txt 格式)。")
        return 0, 0

    # 2. 建立清洗输出目录
    output_base_dir = input_path.parent / "cleaned_output" if input_path.is_file() else input_path / "cleaned_output"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 清洗结果将导出至: {output_base_dir.resolve()}")

    processed_count = 0
    filtered_count = 0

    for file_path in files_to_process:
        # 跳过自身生成的输出目录
        if "cleaned_output" in file_path.parts:
            continue

        print(f"\n📄 正在读取并清洗文件: {file_path.name}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 构建原始 Document
            raw_doc = Document(
                page_content=content,
                metadata={"filename": file_path.name, "source_path": str(file_path.resolve())}
            )

            # 调用流水线清洗
            cleaned_doc = cleaner.clean_document(raw_doc, llm)

            if cleaned_doc is None:
                print(f"🗑️ 文件 {file_path.name} 清洗后字符少于限制或触发黑名单过滤，已舍弃。")
                filtered_count += 1
                continue

            # 输出清洗后的结果
            out_file_path = output_base_dir / f"cleaned_{file_path.name}"
            with open(out_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(cleaned_doc.page_content)
            
            # 显示清洗统计
            original_len = len(content)
            cleaned_len = len(cleaned_doc.page_content)
            reduction = (original_len - cleaned_len) / original_len * 100 if original_len > 0 else 0
            
            print(f"✅ 清洗完成！")
            print(f"   - 原始长度: {original_len} 字符")
            print(f"   - 清洗后长度: {cleaned_len} 字符")
            print(f"   - 杂质缩减比例: {reduction:.2f}%")
            print(f"   - 生成元数据: {cleaned_doc.metadata}")
            processed_count += 1

        except Exception as e:
            print(f"❌ 处理文件 {file_path.name} 出错: {e}", file=sys.stderr)

    return processed_count, filtered_count


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="A2_DocumentCleaner: 生产级 RAG 数据清洗器")
    parser.add_argument("input_path", nargs="?", help="要清洗的文件或文件夹路径")
    parser.add_argument("--config", help="自定义规则配置 (JSON) 路径")
    parser.add_argument("--use_llm", action="store_true", help="是否启用大模型进行指代消解与 OCR 纠错")
    args = parser.parse_args()

    input_path_str = args.input_path
    config_path = args.config
    use_llm = args.use_llm

    print("==============================================")
    print("      RAG 生产级数据清洗管道 - A2_清洗器       ")
    print("==============================================")

    # 1. 交互式获取输入物理文件路径 (不模拟假数据)
    if not input_path_str:
        user_input = input("请输入要清洗的文件或文件夹的绝对/相对路径: ").strip()
        if user_input:
            input_path_str = user_input.strip('"')

    if not input_path_str:
        print("❌ 错误：未指定输入路径，程序结束。")
        sys.exit(1)

    input_path = Path(input_path_str)
    if not input_path.exists():
        print(f"❌ 错误：找不到指定路径: {input_path.resolve()}")
        sys.exit(1)

    # 2. 交互式选择是否开启 LLM 辅助清洗
    if not use_llm and HAS_LANGCHAIN_OPENAI:
        # 检测是否配有环境变量
        if os.getenv("OPENAI_API_KEY"):
            choice = input("是否启用大模型进行【智能纠错】与【指代消解】？(y/n，默认 n): ").strip().lower()
            if choice == "y":
                use_llm = True

    # 3. 初始化 LLM 客户端
    llm = None
    if use_llm:
        if not HAS_LANGCHAIN_OPENAI:
            print("⚠️ 警告：未检测到 langchain_openai 依赖包，将降级为本地规则清洗。")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model_name = os.getenv("MODEL_NAME", "deepseek-chat")
            
            print(f"🤖 正在建立与 {model_name} 的大模型连接...")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                temperature=0.1
            )

    # 4. 初始化清洗管道
    cleaner = RAGDocumentCleaner(config_path=config_path)

    # 5. 执行物理文件批量清洗与落盘
    print(f"\n🚀 开始执行数据清洗工作...")
    processed, filtered = process_file_or_dir(input_path, cleaner, llm)

    print("\n==============================================")
    print(f"✨ 任务执行完毕！")
    print(f"   - 成功清洗并导出文档数: {processed}")
    print(f"   - 被舍弃/过滤的无效文档数: {filtered}")
    print("==============================================")


if __name__ == "__main__":
    main()
