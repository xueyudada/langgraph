# LangChain Core (`langchain_core`) 核心架构指南

`langchain_core` 是整个 LangChain 和 LangGraph 生态的**最底层基石**。它定义了所有的核心接口（Interfaces）和 LCEL（LangChain Expression Language）的运行时逻辑。它**不包含**任何具体的第三方集成（如 OpenAI、Pinecone 等），这也是它极为轻量和稳定的原因。

以下是 `langchain_core` 下最常用、最重要的子包及其核心类与方法结构图：

---

## 1. `langchain_core.runnables` (LCEL 核心)
这是 LangChain 表达式语言 (LCEL) 的引擎。几乎所有核心组件（Model, Prompt, Parser）都继承自 `Runnable`。

*   **`Runnable` (Base Class)**: 所有可运行对象的基类。
    *   `invoke()` / `ainvoke()`: 单次同步/异步执行。
    *   `stream()` / `astream()`: 流式输出。
    *   `batch()` / `abatch()`: 批量执行。
    *   `bind()`: 运行时参数绑定。
    *   `with_fallbacks()`: 配置降级备用策略。
    *   `with_retry()`: 配置重试策略。
*   **`RunnableSequence`**: 核心的“链”（由 `|` 操作符隐式创建）。
*   **`RunnableParallel` / `RunnableMap`**: 并行执行多个 Runnable，常用于分支或组装字典数据。
*   **`RunnableLambda`**: 将任意普通 Python 函数包装成 Runnable。
*   **`@chain`**: 装饰器，将普通函数转换为可被 LCEL 追踪的 Runnable。

---

## 2. `langchain_core.messages` (消息定义)
大模型对话的基础数据结构。

*   **`HumanMessage`**: 用户输入的消息。
*   **`AIMessage`**: 大模型输出的消息。
    *   包含 `content` (文本), `tool_calls` (工具调用请求), `response_metadata` (Token消耗等元数据)。
*   **`SystemMessage`**: 系统提示词。
*   **`ToolMessage`**: 工具执行完毕后，将结果返回给大模型的消息（必须携带 `tool_call_id`）。
*   **`RemoveMessage`**: (LangGraph 中常用) 用于在状态中标记删除某条历史消息。
*   **`AnyMessage`**: 消息类型的 Union，常用于类型提示。
*   **各类 Chunk (`AIMessageChunk` 等)**: 在流式输出 (`stream`) 时返回的消息切片。

---

## 3. `langchain_core.prompts` (提示词工程)
用于动态组装发送给大模型的文本或消息列表。

*   **`ChatPromptTemplate`**: 聊天模型的提示词模板（最常用）。
    *   `from_messages()`: 从消息列表模板初始化。
    *   `from_template()`: 从单段文本初始化。
*   **`PromptTemplate`**: 传统的纯文本大模型提示词模板。
*   **`MessagesPlaceholder`**: 占位符。常用于在 `ChatPromptTemplate` 中动态插入未知长度的历史聊天记录 (`chat_history`) 或代理的便签簿 (`agent_scratchpad`)。

---

## 4. `langchain_core.output_parsers` (输出解析器)
负责将大模型输出的原始文本或消息，转换为程序易于处理的数据结构（如 JSON, Python 对象）。

*   **`StrOutputParser`**: 最常用！将 `AIMessage` 提取为纯文本字符串。流式输出时也非常平滑。
*   **`JsonOutputParser`**: 强制将大模型的文本解析为 Python 字典 (配合提示词使用)。
*   **`PydanticOutputParser`**: 高级解析器。将输出解析为你定义的 Pydantic 模型，并能在解析失败时提供重试指令。
*   **`XMLOutputParser`**: 将输出解析为 XML/字典结构（常用于 Anthropic Claude 等偏好 XML 的模型）。

---

## 5. `langchain_core.tools` (工具与函数调用)
定义大模型可以调用的外部工具接口。

*   **`@tool`**: 极其常用的装饰器！将带有类型注解和 Docstring 的普通 Python 函数一键转化为 LangChain Tool。
*   **`BaseTool`**: 工具的基类。
*   **`StructuredTool`**: 支持复杂 Pydantic 参数架构的工具。
    *   `from_function()`: 将普通函数包装为工具的方法。
*   **`render_text_description()`**: 将工具列表渲染为文本描述，常用于注入到系统提示词中。

---

## 6. `langchain_core.language_models` (模型基类)
定义了所有第三方大模型接入的统一接口。

*   **`BaseChatModel`**: 聊天模型基类（如 `ChatOpenAI`, `ChatAnthropic` 均继承于此）。
    *   核心方法：除了继承自 Runnable 的 `invoke`/`stream` 之外，最重要的特有方法是 `bind_tools()` 和 `with_structured_output()`。
*   **`BaseLLM`**: 传统的文本补全模型基类（目前已较少使用，主推 ChatModel）。

---

## 7. `langchain_core.documents` & `retrievers` (RAG 相关核心)
RAG（检索增强生成）的基础数据结构。

*   **`Document`**: 极度常用的类，表示一条文档。
    *   属性：`page_content` (文档具体文本), `metadata` (来源、页码等元数据的字典)。
*   **`BaseRetriever`**: 检索器基类。
    *   `invoke()`: 输入查询词，返回 `List[Document]`。

---

## 8. 进阶子包 (了解即可)

*   **`langchain_core.embeddings`**: 包含 `Embeddings` 接口，定义了 `embed_documents` 和 `embed_query` 文本向量化标准。
*   **`langchain_core.memory`**: 历史记忆相关的基类接口（注意：随着 LangGraph 的普及，通过状态图管理记忆正成为主流，旧版的 Memory 机制正逐渐边缘化）。
*   **`langchain_core.callbacks`**: 包含了 `BaseCallbackHandler`。这是 LangChain 回调系统的核心，用于实现流式终端打印、日志记录、耗时统计等拦截器功能。
*   **`langchain_core.pydantic_v1`**: 为了兼容旧代码，这里封存了 Pydantic V1 的命名空间（因为 Pydantic V2 变化很大）。
