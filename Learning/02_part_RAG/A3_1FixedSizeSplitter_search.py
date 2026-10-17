import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma

# 导入共享工具
sys.path.append(str(Path(__file__).parent))
from RAG_utils import CustomZhipuEmbeddings

def main():
    load_dotenv()

    print("==============================================")
    print("         RAG 父子块检索查询服务 (A3_1)         ")
    print("==============================================")

    chroma_db_dir = Path(__file__).parent / "chroma_db"
    if not chroma_db_dir.exists():
        print(f"❌ 错误：向量数据库不存在于 {chroma_db_dir.resolve()}，请先运行 A3_1FixedSizeSplitter.py 进行向量化入库。")
        sys.exit(1)

    # 获取智谱 Embedding (读取环境变量)
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "embedding-3")

    if not api_key:
        print("❌ 错误：环境变量中未检测到 EMBEDDINGS_API_KEY，无法向量化。")
        sys.exit(1)

    print(f"🚀 初始化智谱 Embedding 适配器 (模型: {model_name})...")
    embeddings = CustomZhipuEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model_name
    )

    print(f"📦 正在加载本地 Chroma 向量库...")
    vector_store = Chroma(
        persist_directory=str(chroma_db_dir.resolve()),
        embedding_function=embeddings
    )

    while True:
        query = input("\n🔍 请输入您想检索的问题 (输入 'q' 退出，直接回车使用: '博弈模型'): ").strip()
        if query.lower() == 'q':
            print("退出检索服务。")
            break
        if not query:
            query = "博弈模型"

        print(f"\n🔍 正在检索最相似的子块 (top-2)...")
        try:
            results = vector_store.similarity_search_with_relevance_scores(query, k=2)
            
            if not results:
                print("   ⚠️ 未检索到相关匹配项。")
                continue
                
            for i, (doc, score) in enumerate(results):
                print(f"\n🎯 [命中结果 {i+1}] (向量相似度得分: {score:.4f})")
                print(f"   ├─ 检索命中的子块 (Child Chunk) 内容预览:")
                print(f"   │  {doc.page_content.strip()}")
                print(f"   ├─ 对应父块 (Parent ID: {doc.metadata.get('parent_id')[:8]}...) 还原正文:")
                print(f"   │  {doc.metadata.get('parent_content').strip()[:300]}...")
                print(f"   └─ 溯源元数据: {doc.metadata.get('source')}")
                print("-" * 50)
        except Exception as e:
            print(f"❌ 检索失败: {e}")

if __name__ == "__main__":
    main()
