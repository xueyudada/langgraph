import re


def split_by_markdown_header(content : str) -> list[str]:
    """以二级标题（##）或一级标题（#）为边界切分为父块"""
    lines = content.split("\n")#
    chunks = []
    current_chunk = []
    for line in lines:
        if line.startswith("##") or line.startswith("#"):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def split_into_sentence_chunks(text: str,max_size: int =200) -> list[str]:
    """以标点符号（。？！）为边界分句，并组装成100~200字符的子块"""
    # 使用正则按标点符号切分并保留标点
    raw_sentences = re.split(r'(?<=[。！？\n])', text)