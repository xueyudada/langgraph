# LangChain 核心知识点 & 掌握程度清单

> 适用栈:**LangChain 1.x**(本清单基于 1.4 系列)+ `ChatOpenAI`(`langchain-openai`)接 OpenAI 兼容端点
> 总体目标:**熟练用 LCEL / Runnable 把组件串成可用链路**,Agent 层只需了解
> 前置:已完成 LangGraph 基础(本清单中"持久化/记忆""Agent"直接复用)

## 掌握程度图例

| 标记 | 含义 |
|---|---|
| ⭐ 了解 | 知道是什么、何时用、能看懂别人代码 |
| ⭐⭐ 会用 | 能照文档自己写出来、会改参数 |
| ⭐⭐⭐ 熟练 | 不查文档能写、能组合、能排错 |

---

## 前置认知(先建立心智,不用动手)

- LangChain 1.0(2025-10-22)是一次重写:内核收窄,旧功能(LLMChain、SequentialChain、AgentExecutor、旧 Memory)移到 `langchain-classic`,**遇到 `from langchain.chains import LLMChain` 这类老教程要警惕**。
- 重心从"长管道链"转向 **`create_agent` + middleware(架在 LangGraph 上)**。但你的目标是串链,所以 LCEL/Runnable 仍是主战场,Agent 层够用即可。
- 集成包是拆开的:模型走 `langchain-openai`,内核抽象在 `langchain-core`。

---

## 模块 1 · 模型接入层

| 知识点 | 说明 | 目标 |
|---|---|---|
| `ChatOpenAI` + `base_url`/`api_key`/`model` | **你的主入口**。OpenAI 兼容端点全靠它,配 `base_url` 指向 DeepSeek/Qwen/vLLM 等 | ⭐⭐⭐ |
| `init_chat_model("openai:模型名", base_url=...)` | 统一工厂,便于切模型;底层仍是 ChatOpenAI | ⭐⭐ |
| `OpenAIEmbeddings` + `base_url` | 向量化入口,RAG 用;注意 embedding 端点可能与 chat 端点不同 | ⭐⭐ |
| Messages 体系 | `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`,链路的"数据单位" | ⭐⭐⭐ |
| `.content` vs `.content_blocks` | 1.0 新标准化内容块,涉及多模态/推理内容时要懂,避免换模型炸流 | ⭐⭐ |
| `.bind(...)` / `.bind_tools(...)` | 固定 temperature、绑定工具等运行期参数 | ⭐⭐ |
| 调用方式 `invoke/batch/stream` + 异步 `ainvoke/...` | 所有 Runnable 通用,必须形成肌肉记忆 | ⭐⭐⭐ |

> ⚠️ 用 OpenAI 兼容端点时,`with_structured_output`、`bind_tools`、`json mode` 是否可用,**取决于后端模型是否支持 function calling**。先确认你的模型支持 tool calling,再决定结构化输出走哪条路。

---

## 模块 2 · Prompt 与结构化输出

| 知识点 | 说明 | 目标 |
|---|---|---|
| `ChatPromptTemplate.from_messages([...])` | 主力模板写法 | ⭐⭐⭐ |
| `MessagesPlaceholder` | 插入历史消息/agent scratchpad,做多轮必备 | ⭐⭐⭐ |
| `PromptTemplate`(纯字符串) | 老式补全模板,了解即可 | ⭐ |
| **`model.with_structured_output(PydanticModel)`** | **结构化输出首选**,走原生 tool/json,省去手写格式指令 | ⭐⭐⭐ |
| `StrOutputParser` | 把 AIMessage 取成纯文本,链尾常用 | ⭐⭐⭐ |
| `PydanticOutputParser` | **降级为兜底**:只在端点不支持原生结构化输出时用 | ⭐ |
| 其他 parsers(JSON/List 等) | 用到再查 | ⭐ |

---

## 模块 3 · LCEL / Runnable(你的核心战场)

| 知识点 | 说明 | 目标 |
|---|---|---|
| `\|` 管道 与 `.pipe()` | 基础串联,`prompt \| llm \| parser` | ⭐⭐⭐ |
| `RunnableParallel`(或字典写法) | 并行产出多个键,RAG 里"问题+检索上下文"靠它 | ⭐⭐⭐ |
| `RunnablePassthrough` / `.assign()` | 透传 + 往字典里加字段,LCEL 拼装的胶水 | ⭐⭐⭐ |
| `RunnableLambda` | 把普通函数塞进链 | ⭐⭐ |
| `RunnableBranch` | 条件分支路由 | ⭐⭐ |
| `.with_config()` | 传 run name / tags / 配置,调试和追踪用 | ⭐⭐ |
| `.with_retry()` / `.with_fallbacks()` | 重试与降级(主模型挂了切备用) | ⭐⭐ |
| 流式 `stream` / `astream_events` | 链路级流式输出 | ⭐⭐ |

> 这一模块决定你能不能"串出像样的链"。只会 `\|` 串不出真实 RAG,**`RunnableParallel` + `RunnablePassthrough.assign` 才是关键**。

---

## 模块 4 · RAG 栈

| 知识点 | 说明 | 目标 |
|---|---|---|
| Document Loaders | 加载 txt/pdf/web/md 等为 `Document` | ⭐⭐ |
| `RecursiveCharacterTextSplitter` | 最常用切分器,chunk_size/overlap 要会调 | ⭐⭐⭐ |
| Vector Stores(FAISS / Chroma 等) | 本地起步选 FAISS 或 Chroma;`add_documents` / `from_documents` | ⭐⭐ |
| `vectorstore.as_retriever(search_kwargs={...})` | 拿到 retriever,调 `k`、相似度阈值 | ⭐⭐⭐ |
| 手搓检索链(LCEL) | `{"context": retriever \| format_docs, "question": passthrough} \| prompt \| llm` | ⭐⭐⭐ |
| 检索增强的几种范式 | "检索链" vs "把检索做成工具给 agent",知道两条路 | ⭐ |

---

## 模块 5 · Tools / Agent(接 LangGraph,够用即可)

| 知识点 | 说明 | 目标 |
|---|---|---|
| `@tool` 定义工具 | 函数 + docstring + 类型注解 → 工具 | ⭐⭐ |
| `model.bind_tools([...])` | 让模型能发起工具调用(需端点支持 function calling) | ⭐⭐ |
| `create_agent(...)` | 1.0 标准 agent 入口,架在 LangGraph 上,取代 AgentExecutor | ⭐ |
| middleware(`before_model`/`wrap_model_call`/`after_model` 等) | agent 定制方式,知道有这层即可 | ⭐ |
| 记忆/持久化 = LangGraph checkpointer | **不学旧 Memory 类**,直接复用你的 LangGraph 知识 | ⭐⭐(已有) |

---

## 模块 6 · 运行时能力 & 可观测

| 知识点 | 说明 | 目标 |
|---|---|---|
| streaming(`stream`/`astream`/`astream_events`) | 前端逐字输出、拿中间事件 | ⭐⭐ |
| callbacks(如 `UsageMetadataCallbackHandler`) | 统计 token、挂日志;了解机制 | ⭐ |
| LangSmith 追踪 | 设环境变量即可,调链路几乎刚需,能看到每步输入输出 | ⭐⭐ |

---

## 一句话路线

> Messages/content blocks → `ChatOpenAI`(base_url)→ `with_structured_output` → **Runnable 原语 + LCEL** → RAG 栈 → `create_agent`/middleware(了解)→ LangSmith

你清单原有的组件层都在,真正补的是:**`with_structured_output` 替代 parser、Runnable 原语补全、Memory 改指向 checkpointer、加一层 LangSmith**。到这个程度,"用 LCEL 串可用链路"的熟练目标就达成了。
