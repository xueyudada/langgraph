# 大模型 & Agent 全栈学习路线图（项目驱动版）

> 面向「跟着项目代码学」的学习者。每个模块给出：核心知识点、重点标记、掌握程度要求、官方/高星项目链接。
> **本版复核日期：2026-10-17**（上一版 2026-06）。框架与模型迭代极快，版本相关内容一律以官方 README / 文档为准；变更明细见文末「复核记录」。

---

## 图例（先读这一段）

**优先级 / 重点标记**
- 🔴 **核心必学** —— 主干，绕不过去，花最多时间
- 🟡 **重要** —— 提升竞争力，建议掌握
- 🟢 **进阶/选学** —— 看方向和算力，理解原理即可
- ⭐ **桥梁模块** —— 连接两条主线、性价比最高的部分

**掌握程度**
- 【精通】能从零实现 / 理解内部原理 / 能 debug 底层
- 【熟练】能独立用它搭项目、改造、排错
- 【了解】懂概念、知道什么场景用、会跑通官方示例

---

## 一、这是不是两条线？—— 是的，明确分两条

你列的东西本质是**两条独立但有交汇点的主线**：

| | 线 A：Agent 应用开发线 | 线 B：大模型底层线 |
|---|---|---|
| 目标 | 用现成大模型搭出能干活的智能体/应用 | 训练、微调、部署大模型本身 |
| 涉及 | LangChain、LangGraph、单/多 Agent、RAG、上下文工程、MCP、评估、生产部署 | Transformer、数据清洗、预训练、微调、对齐、推理优化 |
| 谁主要学 | AI 应用工程师 / 全栈 | 算法工程师 / 研究员 / 推理部署工程师 |
| 算力门槛 | 低（一张消费级卡甚至 API 即可） | 中到极高（微调要 GPU，预训练要集群） |

**两条线的三个交汇点（桥梁⭐）：**
1. **RAG** —— 用到 embedding 模型、向量检索，是 A 线里最贴近 B 线的部分
2. **微调（Fine-tuning）** —— B 线里性价比最高、应用工程师也该会的部分
3. **推理部署（vLLM）** —— 把微调好的模型部署成 API，再被 A 线的 Agent 调用，形成闭环

**推荐总顺序**（详见第五节）：先打公共基础 → 主攻线 A → 穿插学线 B 的「微调 + 部署」两个桥梁模块 → 想做研究再补线 B 的数据/预训练。

---

## ⚙️ 关于使用 DeepSeek API 学习的统一说明（务必先读）

你打算全程用 DeepSeek API 学习，整体完全可行，且很省钱。下面是**会影响学习路径的几个关键点**，对应模块里也做了标注。

**1. 接入方式（OpenAI 兼容，三行改造）**
- base_url 设为 `https://api.deepseek.com`，用 OpenAI SDK 即可；也支持 **Anthropic 兼容格式**。
- 当前模型（2026/10 复核）：
  - **`deepseek-v4-pro`** —— 旗舰，强推理/编码/Agent，适合复杂任务
  - **`deepseek-v4-flash`** —— 高速低价，适合高频调用、日常学习练手（**学习阶段优先用它，便宜**）
  - 两者都支持 **1M 上下文** 和 **思考/非思考双模式**（`thinking: {"type": "enabled"}` + `reasoning_effort`）
  - ⚠️ 具体模型名、上下文长度、thinking 参数签名以 <https://api-docs.deepseek.com> 为准；升级前先看官方 changelog
- ✅ **旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026/07/24 下线**，新代码直接用 `deepseek-v4-flash` / `deepseek-v4-pro`；看到老教程里写 `deepseek-chat`，替换即可。
- 官方文档：<https://api-docs.deepseek.com>

**2. 🔴 最重要的坑：DeepSeek 没有 Embedding API！**
- 这直接影响 **RAG（A2）**。DeepSeek 只提供对话/推理模型，**不提供向量嵌入接口**，所以做 RAG 时**必须另配一个 embedding 来源**：
  - **推荐（免费、本地跑）**：开源中文 embedding 模型，用 `sentence-transformers` / HuggingFace 本地加载——**BGE-M3**、**bge-large-zh**、**Qwen3-Embedding** 都很强，中文效果好且免费。
  - 或用**第三方 embedding API**：智谱、通义/DashScope、Jina、Voyage 等。
- 结论：**"DeepSeek 负责生成（LLM），embedding 单独找一个"**——这是用 DeepSeek 做 RAG 的标准组合。

**3. 推理模型（思考模式）的两个注意点**
- 多轮对话时，**不要把上一轮的 `reasoning_content`（思维链）塞回历史**，只回传 `content`，否则会报错/串味。
- 思考模式下 `temperature`、`top_p` 等采样参数**不生效**（设了也无效）。
- 好处：`reasoning_content` 能拿到完整 CoT，**学 ReAct / 反思（A5）时拿它当教材特别直观**。

**4. 工具调用 / 函数调用**
- V4 模型**支持** tool calling（A4.1 能正常学），但 OpenAI 兼容 ≠ 行为完全一致，**tool call 的稳定性建议自己测**，复杂 agent 场景多验证几次。

**5. 线 B（大模型底层）不受影响**
- 训练/微调/部署都是针对**开源权重**做的，跟你用哪家 API 无关。而且巧的是 DeepSeek **开源了 V4 权重**、**GRPO 也是它推广的**（见 B3.2），所以它的技术报告和开源模型正好是线 B 的优质学习材料。

---

## 二、公共基础（Part 0）—— 两条线都要先过

### 0.1 Python + 工程基础 🔴【熟练】
- 知识点：Python 进阶（装饰器、异步 async/await、类型注解 typing、Pydantic 数据校验）、虚拟环境（uv / venv）、Git、Docker 基础、REST API 概念、JSON。
- **重点**：`async/await` 和 `Pydantic` —— LangChain/LangGraph/FastAPI 全建立在这两个之上。
- 项目：无需专门项目，边学边用。包管理强烈建议直接上 [`uv`](https://github.com/astral-sh/uv)（生态已全面迁移）。

### 0.2 大模型 API 调用 + Prompt 工程 🔴【熟练】
- 知识点：Chat Completions API（system/user/assistant 角色）、temperature/top_p 等采样参数、token 计费、流式输出（streaming）、function calling（先有概念）、结构化输出（JSON mode）、Prompt 工程（few-shot、CoT 思维链、角色设定、输出格式约束）。
- **DeepSeek 专属要点**：用 OpenAI SDK + `base_url="https://api.deepseek.com"`；练手用 `deepseek-v4-flash`（便宜），复杂任务用 `deepseek-v4-pro`；学会开/关思考模式、读 `reasoning_content`。
- 项目链接：
  - **DeepSeek 官方 API 文档（首要参考，含思考模式/工具调用/JSON/上下文缓存示例）**：<https://api-docs.deepseek.com>
  - Anthropic Cookbook（概念与 Agent 设计通用，可迁移）：<https://github.com/anthropics/anthropic-cookbook>
  - OpenAI Cookbook（接口概念通用）：<https://github.com/openai/openai-cookbook>
  - Prompt 工程指南（高星）：<https://github.com/dair-ai/Prompt-Engineering-Guide>

---

## 三、线 A：Agent 应用开发线（你的主攻方向）

> 学习顺序：A1 → A2 → A3 → A3.5 → A4 → A4.5 → A5 → A5.5 → A6。前后有依赖，别跳。

### A1. LangChain 基础 🔴【熟练】

打基础，理解「组件化」思想。**注意 LangChain 1.0 之后重心明显转向 LangGraph**，所以 LangChain 这块够用即可，不必死磕老版的复杂 Chain。

- 核心知识点：
  - 模型接口：`init_chat_model`、ChatModel、Embeddings
  - **Prompt 模板**、Output Parsers（含 `PydanticOutputParser` 结构化输出）
  - **LCEL（LangChain 表达式语言，`|` 管道语法）**、Runnable 接口 🔴
  - Document Loaders、Text Splitters（为 RAG 铺垫）
  - Vector Stores、Retrievers、Tools、Memory
  - callbacks、streaming
  - **Messages 与 content blocks**（1.0 核心标准）：消息上的 `.content_blocks` 属性提供跨 provider 一致的内容协议，覆盖 reasoning、citations、服务端工具执行等现代能力。不掌握它，切换模型时的流式/多模态很容易踩坑
  - **`with_structured_output()`**：结构化输出的主路径（可用 `PydanticOutputParser` 对照理解底层）
  - **Runnable 原语**：`RunnableParallel` / `RunnablePassthrough` / `RunnableLambda` / `RunnableBranch`，加上 `.bind()`、`.with_config()`。这些才是 LCEL 真正能"串出可用链路"的关键，光会 `|` 串不出像样的 RAG
  - **`create_agent` + middleware**（1.0 头条）：`create_agent` 是新的标准 Agent 入口，跑在 LangGraph runtime 上，取代 `AgentExecutor` 与 `create_react_agent`；通过 `middleware` 数组（`before_model`、`wrap_model_call`、`after_model` 等）做定制。即使只想串链，也该知道"Agent 现在住在哪"
  - **provider 包拆分**：`langchain-openai`、`langchain-anthropic`、`langchain-deepseek` 这类装包/导包的实操，很小但天天用
  - **LangSmith 追踪**：调链路时几乎是刚需，出问题能直接看到每一步的输入输出，比 print 高效太多
- 掌握程度：【熟练】，能用 LCEL 把组件串成一条可用链路即可。
- **DeepSeek 接入**：用官方 `langchain-deepseek` 包的 `ChatDeepSeek`（`pip install langchain-deepseek`），或直接用 `ChatOpenAI` 把 `base_url` 指向 DeepSeek、`model` 设为 `deepseek-v4-flash`/`deepseek-v4-pro`。`init_chat_model("deepseek:deepseek-v4-flash")` 也可。
- 项目链接：
  - 官方主仓库（含示例）：<https://github.com/langchain-ai/langchain>
  - **LangChain 官方文档**（统一文档）：<https://github.com/langchain-ai/docs>
  - **高星开源教程**（LangChain + LangGraph 全套 notebook，适合项目驱动）：<https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial>
  - 经典入门教程：<https://github.com/gkamradt/langchain-tutorials>

### A2. RAG（检索增强生成）🔴⭐【熟练 → 精通】

> ⭐ 桥梁模块。**应用层最值钱的技能之一**，企业落地几乎必用。retrieval 质量决定 RAG 成败。

> ⚠️ **DeepSeek 用户必看**：DeepSeek **没有 embedding 接口**，做 RAG 时 embedding 要单独配。推荐本地跑开源中文模型 **BGE-M3 / bge-large-zh / Qwen3-Embedding**（免费，`sentence-transformers` 或 HuggingFace 加载），生成（LLM）部分继续用 DeepSeek。即：**DeepSeek 出答案 + 开源模型出向量**。reranking 同理可用开源 **bge-reranker** 系列。

- 核心知识点：
  - embedding 原理、相似度（cosine）、embedding 归一化 🔴
  - **文档解析（RAG 的第一痛点，解析错了一切白搭）**：PDF 版面/表格/公式、扫描件 OCR、Excel/Word 结构化提取；工具 **MinerU / docling / unstructured / PaddleOCR** 🔴
  - **数据清洗**：页眉页脚、重复水印、乱码、全半角与标点归一化、空块过滤
  - **chunking 分块策略**：固定长度 / 递归 / 语义分块、overlap 🔴
  - **增量入库与版本管理**：`doc_id` + `content_hash` 去重、按文件变更只重切变动块、删除同步、换 embedding 模型后的全量重建 🔴
  - 向量数据库：FAISS、Chroma、Milvus、pgvector、Qdrant
  - **向量库运维**：HNSW 的 `M` / `ef_construction` / `ef_search`，IVFFlat vs HNSW，召回-延迟-内存三角 🟡
  - 检索进阶：**reranking（cross-encoder 重排）**、hybrid search（BM25 稀疏 + dense 稠密）、query rewriting/expansion、multi-query、parent-document、HyDE 🔴
  - **混合检索工程**：BM25 索引（ES/OpenSearch/pgvector `tsvector`）+ 向量，融合用 **RRF** 或加权；metadata filter 与向量检索的执行顺序 🟡
  - **中文检索细节**：中文 BM25 分词（jieba 类）、停用词、繁简/全半角归一化 🟡
  - **reranker 部署**：bge-reranker 本地跑还是服务化、batch 大小与延迟预算（rerank 10 条 vs 100 条）
  - **长上下文 vs RAG 的决策**：1M 窗口时代，什么时候干脆全文塞进去、什么时候必须检索（别教条）
  - **RAG 评估**：RAGAS 指标（faithfulness 忠实度、answer relevance、context precision/recall）🔴
  - 进阶范式：GraphRAG（知识图谱）、Agentic RAG（与 LangGraph 结合）
- 掌握程度：【熟练】起步，做企业项目要往【精通】走（尤其是检索质量调优 + 评估）。
- 项目链接：
  - **RAG 技巧大全（高星，强烈推荐，几十种技巧逐个 notebook）**：<https://github.com/NirDiamant/RAG_Techniques>
  - 企业级 RAG 引擎（深度文档理解，开箱即用）：<https://github.com/infiniflow/ragflow>
  - LlamaIndex（检索/索引最强的数据框架）：<https://github.com/run-llama/llama_index>
  - RAG 评估框架 RAGAS：<https://github.com/explodinggradients/ragas>
  - RAG 资源/框架导航（社区维护）：<https://github.com/Andrew-Jang/RAGHub>

### A3. LangGraph 🔴【精通】

> 从「线性链」升级到「带状态的图」。这是做正经 Agent 的核心框架，要往精通学。

- 核心知识点：
  - **StateGraph、Node（节点）、Edge（边）** 🔴
  - **State 设计**：TypedDict / Pydantic、reducers（`Annotated` 累加状态）🔴
  - **conditional edges（条件分支）、cycles（循环）** —— LangGraph 相对 LCEL 的核心超能力 🔴
  - **Checkpointer（持久化 / 记忆基础）** 🔴
  - **Human-in-the-loop（`interrupt` 人工介入审批）** 🔴：`interrupt` 的返回值如何恢复执行、`Command(resume=...)` 的用法
  - **`Send`（map-reduce 扇出）**：一个节点动态派发 N 个并行任务再汇聚 —— 批处理/多文档并行处理的基本功 🔴
  - **`Command`**：节点直接返回"下一步去哪 + 状态更新"，比 conditional edge 更适合 Agent 循环
  - **容错**：`RetryPolicy`、节点级超时、失败降级
  - **`durability` 模式**：`exit` / `async` / `sync` 在性能与崩溃恢复粒度之间的取舍
  - **stream 的五种 mode**：`values` / `updates` / `messages` / `custom` / `debug`，前端流式全靠它
  - **Store 的 namespace 设计**：长期记忆按 user / tenant 隔离
  - subgraphs（子图）、time travel / replay（回放调试）
  - LangGraph Studio（可视化调试 IDE）
- 掌握程度：【精通】。State、条件边、Checkpointer 三样必须吃透。
- 项目链接：
  - 官方仓库（含大量 examples）：<https://github.com/langchain-ai/langgraph>
  - **LangChain Academy 官方系统课**（Module 0 环境 → 1-5 进阶 → 6 部署，配 notebook + Studio）🔴：<https://github.com/langchain-ai/langchain-academy>

### A3.5 上下文工程（Context Engineering）🔴⭐【熟练】

> 2025 年后最关键的范式转变：**Agent 的成败不在 prompt 写得多漂亮，而在每一轮往上下文窗口里放什么。**
> 与 A2（该检索什么）、A4.2（该记住什么）有重叠但不相同 —— 它管的是"窗口里此刻该有什么、该扔掉什么"。

- 核心知识点：
  - **上下文预算分配**：system prompt / 工具定义 / 对话历史 / 检索结果 / 工具返回值 各占多少 token 🔴
  - **压缩与摘要**：递归摘要、滚动窗口、结构化笔记（把结论写成清单，而不是保留原始对话）、**tool result 截断与落盘** 🔴
  - **子 Agent 上下文隔离**：sub-agent 只回传结论，不把全过程污染主上下文（这是多 Agent 最实在的价值）
  - **Just-in-time 检索**：让模型自己决定"我要去查什么"（grep / ls / SQL），而不是预先塞一堆
  - **context rot / lost-in-the-middle**：塞得越多反而越差；关键信息要放在首尾
  - **文件系统式记忆**：长任务把状态写进 `notes.md` / 工作目录，按需读回（deepagents、Claude Code 的做法）
  - **缓存友好布局**：稳定内容（system、固定文档）放前缀，配合 A6.4 的 prompt caching
- 掌握程度：【熟练】，能针对一个真实任务画出"每轮窗口里放什么"的表格，并解释为什么这么放。
- 实践建议：拿你 A2 的 RAG 做对照实验 —— 同样 10 个 chunk，分别用「全塞」「先摘要」「按需检索」三种策略，比较准确率与成本。
- 项目链接：
  - Anthropic《Effective context engineering for AI agents》：<https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
  - **deepagents**（文件系统式 memory / 上下文管理，前面已给）：<https://github.com/langchain-ai/deepagents>
  - LangChain 官方文档的 context engineering / memory 章节

### A4. 单 Agent（你特别关注的：工具调用 / 记忆 / Skills / MCP）🔴

#### A4.1 工具调用 Tool Calling / Function Calling 🔴【精通】
- 知识点：function calling 原理、tool schema（JSON Schema）、`@tool` 装饰器、structured output、多工具选择、**parallel tool calls（并行调用）**、tool 错误处理与重试、**ReAct 模式（Reason 推理 + Act 行动循环）** 🔴。
- 掌握程度：【精通】，这是 Agent 的"手脚"。
- **DeepSeek 提示**：V4 模型支持 tool calling，但 OpenAI 兼容 ≠ 行为完全一致，工具调用稳定性建议自己多测；复杂 ReAct 循环里可优先用 `deepseek-v4-pro`。
- 项目链接：
  - LangGraph 官方的 Agent 示例（`create_agent`，`create_react_agent` 已被取代，旧教程仅作读源码参考）
  - **DeepAgents（官方"开箱即用"的 agent 框架，含 planning/子 agent/文件系统，24K+ 星）**：<https://github.com/langchain-ai/deepagents>

#### A4.2 记忆存储 Memory 🔴【熟练】
- 知识点：
  - **短期记忆**（对话历史 / thread，靠 Checkpointer）
  - **长期记忆**（跨会话，靠 Store）🔴
  - 记忆类型：semantic（语义事实）/ episodic（情景）/ procedural（程序性）
  - 记忆的向量化存取与召回、记忆压缩
- 掌握程度：【熟练】。
- 项目链接：
  - **mem0（最流行的 Agent 记忆引擎，高星）**：<https://github.com/mem0ai/mem0>
  - LangGraph 官方 long-term memory store：见 langgraph 文档 memory 章节

#### A4.3 Skills（Agent 技能包）🟡【了解 → 熟练】
- 知识点：Agent Skills 是「可复用能力包」的范式（一个 `SKILL.md` 说明书 + 脚本 + 资源文件），核心思想是 **progressive disclosure（渐进式披露，按需加载，省 token）**；理解它与单个 tool 的区别（skill 是"成套打包的工作流 + 知识"，tool 是"单个函数"）。
- 掌握程度：【了解】概念，能写一个简单 skill 即【熟练】。
- 项目链接：
  - Anthropic 官方 Skills 仓库：<https://github.com/anthropics/skills>
  - 配合 Claude 文档的 Agent Skills 章节阅读。

#### A4.4 MCP（模型上下文协议）🔴【熟练】
- 知识点：
  - MCP 协议本质：基于 **JSON-RPC** 的标准化「LLM ↔ 外部工具/数据」接口
  - **Client / Server 架构** 🔴
  - 三种原语：**Tools（工具）/ Resources（资源）/ Prompts（提示词）** 🔴
  - transport 传输方式：stdio（本地）/ HTTP / SSE
  - **能自己写一个 MCP server** 🔴
  - 在 LangChain/LangGraph 里用 MCP（langchain-mcp-adapters）
- 掌握程度：【熟练】，能写 server + 在 agent 里接入。
- 项目链接：
  - **官方协议规范与文档**：<https://github.com/modelcontextprotocol/modelcontextprotocol> ；文档站 <https://modelcontextprotocol.io>
  - 官方组织（各语言 SDK + 官方 servers）：<https://github.com/modelcontextprotocol>
  - LangChain 的 MCP 适配器：搜索 `langchain-ai/langchain-mcp-adapters`

### A4.5 代码执行、沙箱与浏览器自动化 🟡 → 🔴【熟练】

> 现代 Agent 的"手脚"早已不止 REST 工具：**写代码 → 跑代码 → 看结果**、**操作浏览器**才是主战场。
> 这块直接决定你的 Agent 能不能"真的干活"，也决定它会不会把宿主机搞坏。

- 核心知识点：
  - **沙箱执行**：Docker / Firecracker / gVisor；CPU、内存、时长限制，网络隔离，文件系统只读挂载 🔴
  - **托管沙箱**：E2B、Daytona、Modal —— 免运维，适合学习与小项目
  - **Code interpreter 模式**：模型生成 Python 代码而不是 JSON 工具调用（smolagents 的 `CodeAgent`、deepagents 的文件/命令工具）；优点是组合能力强、步骤少，代价是安全面变大 🔴
  - **浏览器自动化**：Playwright / 浏览器 MCP server、DOM + 截图混合定位、登录态管理、限速与反爬
  - **Computer use**：截图 → 坐标点击的范式，稳定性与成本目前仍偏实验
  - **工具结果治理**：长输出截断、diff 展示、错误信息回灌给模型让它自己修（对应 A3.5）
- 掌握程度：能起一个隔离沙箱让 Agent 跑代码并回传结果，即【熟练】。
- 项目链接：
  - **E2B（开源沙箱，Agent 代码执行标配）**：<https://github.com/e2b-dev/E2B>
  - **smolagents（CodeAgent 范式，~1000 行，读源码最快）**：<https://github.com/huggingface/smolagents>
  - **Playwright（浏览器自动化 / 浏览器 MCP）**：<https://github.com/microsoft/playwright>
  - **OpenHands（开源软件工程 Agent，看工程化怎么落地）**：<https://github.com/All-Hands-AI/OpenHands>

### A5. 多 Agent 协同 + 反思 / 自我纠错 + ReAct 深化 🔴【熟练 → 精通】

- 核心知识点：
  - **多 Agent 架构模式** 🔴：
    - Supervisor（主管调度，一个 manager 分派给专家 agent）
    - Swarm（对等 handoff，agent 之间互相交接）
    - Hierarchical（层级）、Network（网状）
  - **Agent 间通信 / handoff**、A2A（Agent-to-Agent）协议
  - **反思与自我纠错** 🔴：
    - Reflection pattern（生成 → 自我批判 → 修订）
    - **Reflexion**（带记忆的反思）
    - Generator–Evaluator / Critic 循环
  - **推理范式**：ReAct（深化）、Plan-and-Execute（先规划再执行）、Tree of Thoughts
  - 评估与可观测性（接 A6 的 LangSmith）
- 掌握程度：【熟练】，能搭一个 supervisor + 多专家 + 反思循环的系统即达标。
- **DeepSeek 提示**：`deepseek-v4-pro` 思考模式输出的 `reasoning_content`（完整思维链）是学反思/ReAct 的绝佳素材，能直接"看见"模型怎么推理。注意多轮时只回传 `content`、不要把 `reasoning_content` 塞回历史。
- 项目链接：
  - LangGraph 官方多 Agent 教程 + `langgraph-supervisor`、`langgraph-swarm`（见 langchain-ai 组织下仓库）
  - DeepAgents（含子 agent / 规划，前面已给）：<https://github.com/langchain-ai/deepagents>
  - **CrewAI（角色化多 agent 协作，高星，易上手）**：<https://github.com/crewAIInc/crewAI>
  - **AutoGen / AG2（微软系，多 agent 对话编排）**：<https://github.com/microsoft/autogen>
  - **MetaGPT（多 agent 软件公司，高星，看协作设计很有启发）**：<https://github.com/FoundationAgents/MetaGPT>
  - 反思 pattern 参考：LangGraph 官方 `reflection` / `reflexion` 教程

> ⚠️ **别把多 Agent 当默认方案**：能用「单 Agent + 多工具 + 好的状态设计」解决的，就不要拆多 Agent。多 Agent 真正稀缺的价值是**上下文隔离**和**职责边界**，不是"更聪明"。拆之前先问：收益能不能覆盖协调成本与调试难度？

### A5.5 评估体系（Evals）🔴【熟练】

> demo 与生产的分水岭。**没有评测集就调 prompt，等于闭眼改代码。**
> RAG 的指标在 A2（RAGAS）讲；这里讲 **Agent 的评估**，方法论完全不同。

- 核心知识点：
  - **轨迹评估**：不只看最终答案，还要看工具调用序列是否合理、有没有多余/错误步骤、有没有死循环 🔴
  - **数据集构建**：20 条高质量、覆盖真实分布的 case 远胜 2000 条随机样本；必须包含"应该拒答"的负例
  - **LLM-as-judge**：评分 rubric 设计、**judge 校准**（位置偏置、长度偏置、自我偏好）、与人工标注的一致性检验 🔴
  - **指标**：pass@k（能力上限）、**pass^k（稳定性）**、成本、延迟、工具调用准确率
  - **公开基准的定位**：GAIA、τ-bench、BFCL（工具调用）、SWE-bench、terminal-bench —— 用来横向比较模型，不要当业务指标
  - **回归门禁**：改 prompt / 换模型 / 升框架后自动跑，结果 diff 进 CI
  - 与 A6.3 的 trace 打通：线上失败样本自动回流进评测集
- 掌握程度：【熟练】，能给自己 A 线的贯穿项目配一套 ≥30 条 case 的自动评测并接进 CI。
- 项目链接：
  - RAGAS（RAG 侧）：<https://github.com/explodinggradients/ragas>
  - LangSmith 评估文档 / `langsmith` 的 `evaluate()` 接口
  - **DeepEval（pytest 风格，单测 + 评估一体）**：<https://github.com/confident-ai/deepeval>
  - OpenAI Evals（评估设计思路）：<https://github.com/openai/evals>

### A6. 生产部署、安全与工程化 🟡 → 🔴【熟练】

> 想做"能上线的东西"而不止 demo，这块就是 🔴。分 5 小节：部署 → 安全 → 可观测 → 成本 → 产品层。

#### A6.1 服务封装与部署 🔴【熟练】
- **FastAPI** 包成 API；**SSE / WebSocket** 流式返回（LangGraph 的 `astream_events` 直接对接）
- **长任务与幂等**：Agent 一次跑几分钟到几十分钟，必须有任务 ID、断点续跑（checkpointer）、失败重试、**避免重复副作用**
- **队列与背压**：Redis / Celery / ARQ、限流、并发上限；不要让每个请求都直接打模型
- **容器化**：Dockerfile、依赖锁定（uv）、健康检查、冷启动优化
- **部署形态**：**LangGraph Platform / `langgraph-api`（官方主推）**、K8s、Serverless（注意超时限制）
- **CI/CD**：prompt / 模型版本与代码一起进 git，评测集（A5.5）跑通才允许合并

#### A6.2 安全与权限 🔴【熟练】
- **间接提示注入**：RAG 文档、网页、工具返回值、MCP server 返回内容里都可能藏指令 —— 这是最容易被忽视的入口 🔴
- **工具最小权限**：写操作 / 外发 / 删除类工具必须 `interrupt` 人工确认；allowlist 比 blocklist 可靠
- **沙箱执行**：代码与命令执行放容器（见 A4.5），限制网络与文件系统
- **密钥管理**：环境变量 + `.env` 不入库、日志脱敏（**别把 key 拼进 f-string**）
- **MCP 安全**：tool poisoning、confused deputy、只接可信 server、工具描述变更要人工 review
- **数据面**：多租户隔离（向量库按 `tenant_id` 过滤，而不是"记得加 where"）、PII 脱敏、审计日志、数据外泄路径（markdown 图片外链、URL 回带）
- 参考清单：**OWASP LLM Top 10**

#### A6.3 可观测性 🔴【熟练】
- **LangSmith**：tracing、离线数据集、在线监控（最省事）
- 替代 / 补充：**Langfuse**（可自托管）、**Arize Phoenix**、OpenTelemetry GenAI 语义约定（企业统一观测栈时用）
- 关键：trace 要能串起 **LLM 调用 → 工具调用 → 检索命中 → 用户反馈**，否则线上排查等于盲猜

#### A6.4 成本与延迟工程 🟡
- **prompt caching**（DeepSeek 上下文缓存；长 system prompt / 固定文档能省一大笔）
- 模型路由与级联（小模型分流、难例升级）、**semantic cache**、批处理接口
- token 预算与超时；把**每任务成本**当一等指标（不是每 token 成本）

#### A6.5 产品层与数据飞轮 🟡
- 前端：流式渲染、Generative UI、会话与历史管理、鉴权与多租户；（做 Web 端可看 Vercel AI SDK / Next.js）
- 人机协作 UX：审批卡片、编辑后继续、超时兜底
- **数据飞轮**：线上 trace → 失败聚类 → 补进评测集 → 改 prompt / 加工具 / 微调（B3）→ 蒸馏小模型降本
- 反馈采集：点赞/点踩/直接修改稿，比问卷有效得多

**掌握程度**：【熟练】，至少能把一个 LangGraph Agent 用 FastAPI + Docker 部署、接 LangSmith 监控、并通过一次安全自查。

**项目链接**：
- LangSmith + LangGraph 部署官方工作坊：搜索 `langchain-ai/langsmith-agent-lifecycle-workshop`
- LangGraph Platform（部署主推方案）：见 langgraph 官方文档 deployment 章节
- LangServe（旧方案，仅作了解）：搜索 `langchain-ai/langserve`
- LangChain 官方 examples 仓库（含 A2A、部署、评估等实战）：<https://github.com/langchain-samples>
- OWASP LLM Top 10：<https://genai.owasp.org/llm-top-10/>

---

## 四、线 B：大模型底层线

> 📌 **DeepSeek 用户注意**：线 B 全程**和你用哪家 API 无关**——训练、微调、部署都是针对开源权重在自己机器/云上做的。而 DeepSeek 恰好**开源了 V4 等权重**、其技术报告（V3/R1/V4）和 **GRPO** 都是经典学习材料，可以拿它的开源模型当微调/部署的练习对象。

> 学习顺序：B0 →（B1）→（B2）→ **B3 ⭐ → B4 ⭐**。如果时间有限、目标是应用工程师，**B0 + B3 + B4 是必学；B1/B2 只需【了解】即可**（真要做算法岗再往【熟练】走）。

### B0. 深度学习 & Transformer 基础 🔴【精通】

> B 线的根。这块不牢，后面全是黑箱。

- 核心知识点：
  - **前置数学（半天即可）**：矩阵乘法的形状直觉、softmax 的数值稳定性、交叉熵 = 负对数似然、梯度下降 —— 不求推导，求"看到公式不慌"
  - PyTorch：张量、自动微分（autograd）、训练循环、Dataset/DataLoader
  - **注意力机制**：self-attention、multi-head attention、causal mask 🔴
  - **Transformer 架构**：encoder/decoder、FFN、residual、LayerNorm / RMSNorm 🔴
  - 位置编码：绝对位置 → **RoPE（旋转位置编码，现代主流）** 🔴
  - **Tokenization**：BPE 分词、词表、embedding 🔴
  - 训练要素：softmax、cross-entropy 损失、AdamW 优化器、learning rate schedule
- 掌握程度：【精通】，目标是"能从零手写一个 GPT-2"。
- 项目链接（**Karpathy 全家桶，项目驱动学习的黄金标准**）：
  - **build-nanogpt（从空文件一步步搭出 GPT-2，配 YouTube 视频，commit 按步骤拆解）** 🔴：<https://github.com/karpathy/build-nanogpt>
  - nanoGPT（精简训练框架，已被 nanochat 取代但仍是经典）：<https://github.com/karpathy/nanoGPT>
  - llm.c（纯 C/CUDA 实现，想深入底层/CUDA 看这个）：<https://github.com/karpathy/llm.c>
  - 进阶系统课 Stanford CS336《从零构建 LLM》：<https://stanford-cs336.github.io>

### B1. 数据清洗与处理 🟡【了解 → 熟练】

> 应用工程师只需知道"数据质量决定模型上限"以及管线长什么样；打算做算法/预训练时再往【熟练】走。

- 核心知识点：
  - 数据来源：Common Crawl、FineWeb 等
  - **去重**：exact dedup、MinHash / 模糊去重 🔴
  - **质量过滤**：启发式规则 + 分类器过滤 🔴
  - 语言识别、PII（隐私信息）去除、毒性过滤
  - **去污染 decontamination**（避免测试集泄漏）
  - 数据配比、instruction/chat 格式整理（为微调准备）
- 掌握程度：【了解】能看懂一条清洗管线在做什么；要【熟练】则需亲手跑通 datatrove 的一条管线。
- 项目链接：
  - **datatrove（HuggingFace 大规模数据处理库，FineWeb 用的就是它）**：<https://github.com/huggingface/datatrove>
  - FineWeb 数据集 + 技术报告（怎么造高质量数据的标杆）：HuggingFace 上搜 `HuggingFaceFW/fineweb`

### B2. 预训练 Pretraining 🟢 → 🟡【了解原理 + 能跑小规模】

> 算力门槛最高。除非有集群，否则**重在理解 + 跑 nanochat 级别的小复现**，别硬刚大模型。

- 核心知识点：
  - 因果语言建模目标（next-token prediction）
  - **分布式训练**：DDP、**FSDP**、DeepSpeed、Megatron；数据并行 / 张量并行 / 流水线并行 🟡
  - 混合精度（bf16 / fp8）、gradient checkpointing
  - **Scaling Laws（缩放定律）** 🟡、compute-optimal
  - lr schedule（warmup + cosine）、batch size、训练监控（loss / grad norm）
- 掌握程度：【了解】原理；能用 nanochat 跑通一个 mini 模型即【熟练】。
- 项目链接：
  - **nanochat（Karpathy 最新作，~$100 / 4 小时复现一个能对话的 ChatGPT，覆盖 分词→预训练→中训→SFT→RL→推理，约 8000 行干净代码）** 🔴：<https://github.com/karpathy/nanochat>
  - build-nanogpt / llm.c（同上，预训练视角）

### B3. 微调 Fine-tuning 🔴⭐【熟练】

> ⭐⭐ **整个 B 线性价比最高、应用工程师最该会的部分。** 能把通用模型变成你领域的专用模型。

#### B3.1 SFT 监督微调 🔴【熟练】
- 核心知识点：
  - instruction tuning、**chat template（训练/推理必须一致）** 🔴
  - 全参微调 vs **PEFT 参数高效微调** 🔴
  - **LoRA / QLoRA 原理**（低秩分解，rank / alpha 超参）🔴、DoRA
  - **量化**：4-bit / 8-bit、bitsandbytes、GPTQ / AWQ 🔴
  - 数据格式：alpaca / sharegpt / OpenAI messages
  - **合成数据与蒸馏**：用强模型生成领域 SFT 样本（self-instruct / 拒绝采样），**质量过滤比数量重要** 🔴
  - **怎么评估微调结果**：留出集 + 与 base 模型 A/B、防灾难性遗忘、把评估接进 A5.5（不自评 = 没微调）
  - 训练技巧：packing、NEFTune、超参（lr、epochs）
- 掌握程度：【熟练】，目标"能独立微调出一个领域模型并评估"。
- 项目链接：
  - **LLaMA-Factory（最流行，零代码 WebUI/LlamaBoard，支持 100+ 模型，~68K 星，新手首选）** 🔴：<https://github.com/hiyouga/LLaMA-Factory>
  - **Unsloth（训练快 2x、省 70% 显存，免费 Colab notebook，单卡党福音）** 🔴：<https://github.com/unslothai/unsloth>
  - **TRL（HuggingFace 官方，生态最正统，SFT/DPO/GRPO 全有）**：<https://github.com/huggingface/trl>
  - Axolotl（配置驱动，偏生产）：<https://github.com/axolotl-ai-cloud/axolotl>

#### B3.2 对齐 / 偏好学习 Alignment 🟡【了解 → 熟练】
- 核心知识点：
  - **RLHF**（reward model + PPO）—— 经典但复杂
  - **DPO（直接偏好优化，免 reward model，更稳）** 🔴
  - **GRPO（DeepSeek 推广，免 critic，当下推理模型对齐主流）** 🔴
  - KTO、ORPO、SimPO
  - reasoning 模型的 RL 训练（R1 风格）
- 掌握程度：【了解】原理 + 能用 TRL 跑通一个 DPO/GRPO 即【熟练】。
- 项目链接：
  - TRL（DPO/GRPO/PPO 参考实现）：<https://github.com/huggingface/trl>
  - LLaMA-Factory / Unsloth 也都支持 DPO/GRPO（直接复用上面的链接）

#### B3.3 数据合成与蒸馏 🟡【了解 → 熟练】
- **合成数据**：self-instruct、Evol-Instruct、拒绝采样（强模型生成多样本，再按规则/校验器过滤）
- **推理蒸馏**：教师模型生成"题目 + 解题过程 + 答案"，训练小模型（DeepSeek-R1 蒸馏系列是标杆）
- **质量治理**：去重、难度分层、答案可验证性（数学/代码优先用可执行程序校验）
- ⚠️ 合成数据的分布偏差会被放大；必须留独立测试集，**别用合成集自测**

### B4. 模型部署与推理优化 Serving 🔴⭐【熟练】

> ⭐ 桥梁模块。把你微调好的模型部署成 API，再交给线 A 的 Agent 调用 = 全栈闭环。

- 核心知识点：
  - **推理引擎**：**vLLM（PagedAttention + continuous batching，最主流）** 🔴、SGLang、TGI；本地/CPU 用 **llama.cpp / Ollama** 🔴
  - **量化部署**：GPTQ / AWQ / GGUF / FP8
  - 关键机制：**KV cache、PagedAttention、continuous batching、speculative decoding（投机解码）** 🔴
  - **OpenAI 兼容 API server**、tool calling parser（让本地模型也能函数调用）🔴
  - 多 GPU：tensor / pipeline parallelism
  - 性能指标：**TTFT（首 token 延迟）、TPOT（每 token 时间）、throughput（吞吐）** 🔴
  - **结构化输出约束**：guided decoding / grammar-constrained decoding（vLLM `guided_json`、Outlines、XGrammar），让本地模型稳定吐 JSON 🟡
  - **多 LoRA 热加载**：一个 base 模型挂多个 adapter，按请求路由（多租户 / 多任务）🟡
- 掌握程度：【熟练】，能把模型用 vLLM 起一个 OpenAI 兼容服务并被 agent 调用。
- 项目链接：
  - **vLLM（官方，最主流推理引擎，2000+ 贡献者）** 🔴：<https://github.com/vllm-project/vllm>
  - **Ollama（本地一键跑模型，最易上手）**：<https://github.com/ollama/ollama>
  - **llama.cpp（CPU / 量化 / 边缘部署之王）**：<https://github.com/ggml-org/llama.cpp>
  - TGI（HuggingFace 推理服务）：<https://github.com/huggingface/text-generation-inference>

---

## 五、推荐总学习顺序（把两条线编织起来）

下面是综合两条线、兼顾"先学 Agent"诉求的推荐推进顺序。**横向并行的部分可以穿插**。

```
阶段 0  公共基础
        Python/异步/Pydantic  →  大模型 API + Prompt 工程
                                   │
阶段 1  线 A 主干（核心投入）        │
        LangChain 基础(A1)
            → RAG(A2) ⭐ ←──────────┐   （此处开始接触 embedding，轻触线 B）
            → LangGraph(A3)         │
            → 上下文工程(A3.5) ⭐    │
            → 单 Agent(A4: 工具/记忆/Skills/MCP)
            → 代码执行/沙箱(A4.5)    │
            → 多 Agent + 反思/ReAct(A5)
            → 评估体系(A5.5) 🔴      │
            → 生产部署/安全/监控(A6)
                                    │
阶段 2  线 B 桥梁（强烈建议穿插，最高性价比）
        Transformer 基础(B0) ⭐ ────┘   （手写一个 nanoGPT，理解模型在干嘛）
            → 微调 SFT/LoRA(B3) ⭐         （LLaMA-Factory / Unsloth 跑通）
            → 推理部署 vLLM(B4) ⭐         （部署成 API → 回接阶段 1 的 Agent）
            → 对齐 DPO/GRPO(B3.2)
                                    │
阶段 3  线 B 深水区（想做研究/算法再上）
        数据清洗(B1)  →  预训练 + 分布式(B2)  →  nanochat 全流程复现
```

**为什么这样排：**
- 你明确想先学 Agent，所以**阶段 1 全程走线 A**，能最快产出可用东西、建立成就感。
- **阶段 2 的 B0 + B3 + B4 是三个⭐桥梁**：理解模型 → 微调模型 → 部署模型，正好补上"只会调 API"的短板，且和线 A 形成闭环（自己微调的模型用 vLLM 部署后，给 LangGraph agent 调用）。
- **阶段 3（数据清洗、预训练、分布式）算力和理论门槛最高**，是研究/算法岗的深水区。普通应用工程师"了解原理 + 用 nanochat 跑一遍小复现"即可，不必一开始就陷进去。

**每阶段的时长与验收标准（DoD）**——项目驱动最容易"跑通了但不会"，用这张表卡进度：

| 阶段 | 建议时长 | 产出物 | 验收标准 |
|---|---|---|---|
| 0 公共基础 | 1–2 周 | 一个能流式输出 + 结构化 JSON 的 CLI 脚本 | 能解释 temperature / top_p 对输出的影响；能写出带 `Field` 约束的 Pydantic 输出模型 |
| 1 线 A 主干 | 6–10 周 | 贯穿项目 v1（RAG + 工具 + 记忆 + 部署） | 自有 ≥30 条评测集可自动跑；检索命中可溯源到文件与 chunk；Agent 有步数/token 上限且不会死循环 |
| 2 线 B 桥梁 | 3–6 周 | 一个 LoRA 微调模型 + vLLM 服务 | 微调后在留出集上优于 base；vLLM 的 OpenAI 兼容端点能被阶段 1 的 Agent 直接调用 |
| 3 线 B 深水区 | 按需 | nanochat 最小档复现 | 能说清数据配比、并行策略、scaling law 对你这次实验的影响 |

---

## 六、各模块掌握程度 & 优先级速查表

| 模块 | 线 | 优先级 | 掌握程度 | 代表项目 |
|---|---|---|---|---|
| Python/异步/Pydantic | 公共 | 🔴 | 熟练 | uv |
| 大模型 API + Prompt | 公共 | 🔴 | 熟练 | anthropic-cookbook |
| A1 LangChain 基础 | A | 🔴 | 熟练 | langchain / LangChain-OpenTutorial |
| A2 RAG | A | 🔴⭐ | 熟练→精通 | RAG_Techniques / ragflow |
| A3 LangGraph | A | 🔴 | 精通 | langgraph / langchain-academy |
| A3.5 上下文工程 | A | 🔴⭐ | 熟练 | deepagents / Anthropic 文章 |
| A4.1 工具调用 | A | 🔴 | 精通 | deepagents |
| A4.2 记忆 | A | 🔴 | 熟练 | mem0 |
| A4.3 Skills | A | 🟡 | 了解→熟练 | anthropics/skills |
| A4.4 MCP | A | 🔴 | 熟练 | modelcontextprotocol |
| A4.5 代码执行/沙箱 | A | 🟡→🔴 | 熟练 | E2B / smolagents / OpenHands |
| A5 多 Agent + 反思 | A | 🔴 | 熟练→精通 | CrewAI / AutoGen / MetaGPT |
| A5.5 评估体系 | A | 🔴 | 熟练 | RAGAS / DeepEval / LangSmith |
| A6 部署/安全/监控 | A | 🟡→🔴 | 熟练 | LangSmith / LangGraph Platform |
| B0 Transformer 基础 | B | 🔴⭐ | 精通 | build-nanogpt / llm.c |
| B1 数据清洗 | B | 🟡 | 了解→熟练 | datatrove / fineweb |
| B2 预训练 | B | 🟢→🟡 | 了解+小复现 | nanochat |
| B3.1 微调 SFT/LoRA | B | 🔴⭐ | 熟练 | LLaMA-Factory / Unsloth |
| B3.2 对齐 DPO/GRPO | B | 🟡 | 了解→熟练 | TRL |
| B3.3 数据合成与蒸馏 | B | 🟡 | 了解→熟练 | TRL / R1 蒸馏 |
| B4 推理部署 vLLM | B | 🔴⭐ | 熟练 | vLLM / Ollama / llama.cpp |

---

## 七、补充：综合性课程（项目驱动者额外推荐）

- **HuggingFace Agents Course（免费，含 smolagents / LlamaIndex / LangGraph 三框架 + 评估 + 微调，带证书和 leaderboard）**：<https://github.com/huggingface/agents-course>
- **smolagents（HuggingFace 极简 agent 框架，~1000 行，CodeAgent 直接写 Python 代码当动作，读源码学 agent 本质特别好）**：<https://github.com/huggingface/smolagents>
- **Anthropic Cookbook（Claude 工具调用 / RAG / Agent 实战代码）**：<https://github.com/anthropics/anthropic-cookbook>

---

## 八、给"项目驱动学习者"的方法建议

1. **每个模块都"跑通 → 读源码 → 改造 → 复现"四步走**，别只看不动手。Karpathy 系列和 LangChain Academy 的 commit/notebook 都是按步骤拆的，照着走。
2. **先广后深**：先用 1-2 周把线 A 主干快速跑一遍（哪怕粗糙），建立全局认知，再回头逐模块深挖。
3. **做一个贯穿项目**：比如「一个能查你私有文档(RAG) + 调用工具(Tool/MCP) + 带记忆 + 评估 + 部署上线」的助手，把 A 线模块全串起来；多 Agent 作为进阶项（单 Agent 先跑通，见 A5 的提醒）；再进阶用自己微调(B3)、vLLM 部署(B4)的模型替换掉 API。
4. **框架迭代极快**（LangChain/LangGraph 几乎月更），**永远以官方仓库最新 README + docs 为准**，教程类资源看思想即可，代码 API 可能已变。
5. 算力不够时：微调用 Unsloth 的免费 Colab；部署用 Ollama 本地跑量化模型；预训练用 nanochat 的最小档（~$15-100 租云 GPU）。

---

## 九、决策速查（选型时先查这张表）

| 你的情况 | 选择 | 理由 |
|---|---|---|
| 文档总量小、稳定、能塞进窗口 | **长上下文直接塞** | 少一层检索误差，实现最简单 |
| 文档量大 / 更新频繁 / 需权限过滤 / 要溯源 | **RAG** | 成本、实时性、可审计 |
| 需要跨文档综合、多跳推理 | **Agentic RAG / GraphRAG** | 单轮检索补不全 |
| 行为、语气、格式不对 | **先改 prompt + few-shot** | 成本最低，见效最快 |
| 缺领域知识且无法检索（内网、私有术语） | **微调（LoRA）** | 把知识内化 |
| 要固定输出格式 / 低延迟 / 小模型 | **微调** | 比 prompt 稳定 |
| 流程线性、≤5 步、无分支 | **直接函数 + LCEL** | 别上框架 |
| 有分支、循环、人工审批、需持久化 | **LangGraph** | 状态与恢复是刚需 |
| 子任务彼此独立、上下文会爆 | **多 Agent / sub-agent** | 上下文隔离是真实收益 |
| 只是"多调一个工具" | **单 Agent + 多工具** | 多 Agent 属于过度设计 |
| 需要跑代码 / 改文件 / 操作浏览器 | **沙箱 + code interpreter / browser agent** | 安全与能力（A4.5） |
| 合规 / 审计要求高 | **可自托管栈**（Langfuse、私有向量库、本地模型） | 数据不出域 |

---

## 十、反模式清单（比"该学什么"更容易踩的坑）

1. **没有评测集就调 prompt** —— 你改的每一次可能都在变差，而且看不出来。
2. **prompt 越堆越长** —— 先做上下文工程（A3.5），不是继续加字。
3. **过早多 Agent** —— 协调成本、调试难度、token 成本同时上升。
4. **把所有历史塞进上下文** —— 触发 context rot，又贵又慢。
5. **工具描述写成散文** —— 模型选错工具，十有八九是描述问题，不是模型问题。
6. **Agent 循环没有预算上限** —— 必须有 max steps / token 上限 / 循环检测。
7. **密钥写进代码或日志** —— 从第一天就用环境变量。
8. **无确认、无幂等的副作用操作** —— 发邮件、删数据、下单必须人工确认 + 幂等键。
9. **检索结果不看就上** —— 没有 rerank、没有溯源、没有 hit-rate 观测的 RAG 等于抽奖。
10. **只在 demo 数据上调优** —— 真实分布的脏数据会让你前面所有优化归零。

---

## 十一、术语表

| 术语 | 一句话解释 |
|---|---|
| Token | 模型处理文本的最小单位；计费与上下文长度都按它算 |
| Embedding | 把文本映射成向量，用于相似度检索 |
| Chunk | 文档切分后的片段，是检索的最小单位 |
| Rerank | 用 cross-encoder 对召回结果重新排序，显著提升精度 |
| RRF | Reciprocal Rank Fusion，多路召回结果的融合方法 |
| Agent | 能"思考 → 调工具 → 观察 → 再思考"循环直到完成任务的系统 |
| Tool calling | 模型按 schema 输出结构化调用请求，由外部执行并回灌结果 |
| ReAct | Reason + Act 的交替循环范式 |
| HITL | Human-in-the-loop，在关键步骤插入人工审批 |
| Checkpointer | LangGraph 的状态快照存储，是记忆、断点续跑、时间旅行的基础 |
| Context engineering | 决定每一轮往上下文窗口里放什么、扔什么 |
| MCP | Model Context Protocol，LLM ↔ 工具/数据的标准协议（JSON-RPC） |
| A2A | Agent-to-Agent 协议，Agent 之间的互操作 |
| Eval | 用固定数据集 + 指标衡量系统质量，Agent 还要评估轨迹 |
| Trace | 一次请求的完整调用链路记录（LLM、工具、检索） |
| LoRA | 低秩适配微调，只训练少量参数 |
| DPO / GRPO | 偏好对齐方法；GRPO 由 DeepSeek 推广，免 critic |
| vLLM | 主流推理引擎，核心是 PagedAttention + continuous batching |
| TTFT / TPOT | 首 token 延迟 / 每 token 生成时间，是用户体感的核心指标 |

---

## 十二、复核记录（Changelog）

| 日期 | 变更 |
|---|---|
| 2026-10-17 | ① 修复 A1 知识点排版（句子粘连、标点半角）；② DeepSeek 模型信息改为「2026/10 复核」，`deepseek-chat`/`deepseek-reasoner` 改为「已于 2026/07/24 下线」；③ `create_react_agent` → `create_agent`，LangServe 降级为旧方案；④ B1 优先级改为 🟡 并消除与第四节的矛盾；⑤ A2 补：文档解析、数据清洗、增量入库与版本管理、向量库运维、混合检索工程（RRF）、中文检索、reranker 部署、长上下文 vs RAG 决策；⑥ A3 补：`Send`、`Command`、`RetryPolicy`、`durability`、stream 五种 mode、Store namespace、`interrupt` 恢复；⑦ 新增 **A3.5 上下文工程**、**A4.5 代码执行/沙箱/浏览器**、**A5.5 评估体系**；⑧ A6 拆为部署/安全/可观测/成本/产品层五小节；⑨ B0 补数学前置、B3.1 补合成数据与微调评估、新增 B3.3 数据合成与蒸馏、B4 补结构化输出约束与多 LoRA；⑩ 第五节增加阶段时长与 DoD 表；⑪ 新增决策速查表、反模式清单、术语表 |
| 2026-06 | 初版 |

---

> 路线会随生态变化，建议每隔几个月对照官方仓库 star 数和 release 复核一次工具选型；**版本相关的内容一律以官方文档为准**。祝学习顺利。
