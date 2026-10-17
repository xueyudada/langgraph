import os.path

from langchain_community.document_loaders import TextLoader


def main():
    print("=== 1. TextLoader (单文本加载器) ===")
    textLoader = TextLoader(os.path.join(".","test.txt"),encoding="utf-8")
    docs = textLoader.load()

    print(f"加载文档数量:{len(docs)}")
    print(f"文档内容:\n{docs[0].page_content}")
    print(f"元数据(Metadata):{docs[0].metadata}")
    print("-"*50)

if __name__ == "__main__":
    main()