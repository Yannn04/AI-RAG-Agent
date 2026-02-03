# 智扫通 (AI RAG Agent) 🧠🤖
扫地机器人商品客服，基于RAG技术知识库检索智能客服问答、生成用户报告与优化建议。


**简介**

本项目实现了一个基于 RAG（Retrieval-Augmented Generation）思路的智能客服原型，采用 Chroma 向量数据库做向量检索，结合可配置的 LLM 聊天模型与嵌入模型，为中文场景提供问答与总结能力。前端使用 `streamlit` 提供交互式 Web UI（`app.py`）。

---

## ⭐ 主要特性

- 基于 **RAG** 的检索增强生成流程（`rag/`）
- 使用 **Chroma** 做向量存储（持久化目录由 `config/chroma.yml` 配置）
- 支持 **txt / pdf** 等文件加载并自动去重（通过 MD5）
- 可配置的提示词体系（`prompts/` + `config/prompts.yml`）
- 模型与嵌入由 `config/rag.yml` 配置（支持 `langchain_community` 中的模型）
- 可扩展的 agent tools 与 middleware（`agent/tools/`）
- 简洁的 Streamlit UI（`streamlit run app.py`）

---

## 📁 项目结构（精要）

- `app.py` — Streamlit 前端入口
- `agent/` — Agent 逻辑与工具（`react_agent.py`, `tools/` 等）
- `rag/` — Vector store 与 RAG 服务（`vector_store.py`, `rag_service.py`）
- `config/` — YAML 配置（`chroma.yml`, `rag.yml`, `prompts.yml`, `agent.yml`）
- `prompts/` — 系统、RAG、报告等提示词模板
- `data/` — 待导入的知识文件（`*.txt`, `*.pdf`），以及 `external/` 数据
- `logs/` — 运行日志
- `model/` — 模型工厂（将配置的模型实例化）
- `utils/` — 配置、文件加载、日志、路径等工具

---

## 🚀 快速开始

1. 推荐 Python 版本：**Python 3.10+**，并创建虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

2. 安装依赖（仓库无统一 `requirements.txt` 时可手动安装）：

```bash
pip install streamlit langchain-core langchain-chroma langchain-text-splitters langchain-community pyyaml
# 如需 chroma 原生支持，可额外安装 chromadb
pip install chromadb
```

3. 配置模型与检索参数：编辑 `config/rag.yml` 和 `config/chroma.yml`（例如：`chat_model_name`, `embedding_model_name`, `collection_name`, `persist_directory` 等）以及 `prompts/*.txt`。

4. 准备知识数据：将 `*.txt` 或 `*.pdf` 放到 `data/`（`config/chroma.yml` 中 `data_path` 所指目录）。

5. 将数据加载到向量库（增量加载，会基于 `md5.text` 跳过已加载的文件）：

```bash
python rag/vector_store.py
```

6. 启动前端（演示界面）：

```bash
streamlit run app.py
```

7. 调试/演示：

```bash
# 测试 RAG 总结服务
python rag/rag_service.py

# 或执行 Agent 流式输出示例
python agent/react_agent.py
```

---

## ⚙️ 配置说明

- `config/chroma.yml`:
  - `collection_name`: 向量集合名
  - `persist_directory`: Chroma 数据持久化目录（默认 `chroma_db`）
  - `k`: 检索返回的文档数量
  - `chunk_size`, `chunk_overlap`, `separators`: 文本分片配置
  - `data_path`: 知识文件目录
  - `md5_hex_store`: MD5 去重记录文件

- `config/rag.yml`:
  - `chat_model_name`, `embedding_model_name`: 在 `model/factory.py` 中引用

- `config/prompts.yml`: 指向 `prompts/` 下的提示词文件路径

---

## 🛠️ 开发说明

- 添加/扩展工具：在 `agent/tools/agent_tools.py` 中定义函数并在 `agent/react_agent.py` 中注册。
- 日志：使用 `utils/logger_handler.py`，日志文件保存在 `logs/`。
- 重新构建向量库：如需清空重建，请删除 `chroma_db/` 和 `md5.text`，再运行 `python rag/vector_store.py`。

---

## ❓ 常见问题（快速排查）

- 找不到 `streamlit` / 运行报错：确认虚拟环境已激活并安装依赖。
- 向量加载无效果：检查 `data/` 下文件类型是否被 `config/chroma.yml` 的 `allow_knowledge_file_type` 包含，查看 `logs/` 获取详细信息。

---



