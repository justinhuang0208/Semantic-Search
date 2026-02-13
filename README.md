# Semantic Search (OpenRouter / Ollama Embedding)

本專案提供本機優先的語意搜尋系統，目標資料是卡片式 Markdown。

- Embedding provider: OpenRouter or local Ollama
- Default model (OpenRouter): `google/gemini-embedding-001`
- Default model (Ollama): `qwen3-embedding:0.6b`
- Retrieval: Hybrid (`BM25 + Vector`) + RRF
- Storage: SQLite + FAISS
- Interface: CLI

## 1. 建立 Anaconda 環境

```bash
conda env create -f environment.yml
conda activate semsearch
```

## 2. 安裝專案

```bash
pip install -e .
```

## 3. 設定 API Key（OpenRouter 模式）

```bash
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_API_KEY>"
```

若使用本地 Ollama 模式（`--use-local-embedding`），不需要設定 `OPENROUTER_API_KEY`。

## 4. 設定待嵌入資料夾（可選）

可用環境變數設定 `ingest` 的預設來源路徑：

```bash
export SEMSEARCH_SOURCE="/你的/Markdown/資料夾"
```

若同時有帶 `--source`，會以 `--source` 為準。

## 5. 建立索引

```bash
semsearch ingest --source "1 - Cards" --rebuild
```

輸出內容包含：文件數、chunk 數量、embedding 維度、SQLite 路徑與 FAISS 路徑。

若已設定 `SEMSEARCH_SOURCE`，可省略 `--source`：

```bash
semsearch ingest --rebuild
```

增量更新（推薦日常使用）：

```bash
semsearch ingest --source "1 - Cards"
```

本地 Ollama 模式建立索引：

```bash
ollama pull qwen3-embedding:0.6b
ollama serve
semsearch ingest --source "1 - Cards" --use-local-embedding --rebuild
```

- 不帶 `--rebuild` 時，系統會比較檔案內容 hash，只更新新增/變更檔案，並刪除索引中已不存在的檔案。
- 系統會使用 embedding 快取（`content_hash + model`）避免重複呼叫 API。
- 即使使用 `--rebuild`，只要內容 hash 曾經快取過，仍會直接重用，不會再次上傳到 embedding API。

### Source 與資料庫路徑的關係

- `--source` 只決定「這次 ingest 要讀哪個資料夾」。
- 是否為不同資料庫，取決於 `--db-path` 與 `--faiss-path`，不是 `--source`。
- 若只更換 `--source`、但沿用同一組 `--db-path/--faiss-path`，會覆蓋同一套索引（看不到的舊文件會被當成刪除）。

範例：建立兩套獨立索引（A 與 B）

```bash
semsearch ingest --source "/path/A" --db-path data_index/A.db --faiss-path data_index/A.faiss
semsearch ingest --source "/path/B" --db-path data_index/B.db --faiss-path data_index/B.faiss
```

## 6. 查詢

```bash
semsearch query "non-blocking assignment 在跨週期傳遞的重點" --top-k 8
semsearch query "MULH sign extension 高位錯誤" --top-k 8 --show-chunk-type
```

本地 Ollama 模式查詢：

```bash
semsearch query "non-blocking assignment 在跨週期傳遞的重點" --use-local-embedding --top-k 8
```

## 7. 評估

```bash
semsearch eval --golden tests/golden_queries.yaml
semsearch eval --golden tests/golden_queries.yaml --verbose
semsearch eval --golden tests/golden_queries.yaml --use-local-embedding
```

## 8. Embedding 模式與參數規則

- `--use-local-embedding` 會切換到本地 Ollama（`http://localhost:11434/api/embed`）。
- `--model` 可覆寫預設模型：
  - 未加 `--use-local-embedding`：預設 `google/gemini-embedding-001`
  - 加上 `--use-local-embedding`：預設 `qwen3-embedding:0.6b`
- `ingest/query/eval` 需要使用與索引建立時一致的 provider/model。
- 若設定不一致，系統會報錯並提示先用正確參數重新 `ingest`。

## 9. 專案結構

- `semsearch/cli.py`: CLI 入口（`ingest`, `query`, `eval`）
- `semsearch/pipeline.py`: ingest/query/eval 主流程
- `semsearch/markdown_ingest.py`: Markdown 解析與切塊
- `semsearch/embeddings.py`: OpenRouter/Ollama embedding client + provider resolver
- `semsearch/storage.py`: SQLite schema 與資料存取
- `semsearch/vector_index.py`: FAISS 建索引與查詢
- `semsearch/retrieval.py`: BM25、RRF、結果去重
- `tests/golden_queries.yaml`: 第一版 golden set

## 10. Chunk 策略

- 檔案 `< 1200` 字：整篇一塊
- 檔案 `1200 ~ 3500` 字：按 `##` 區段切
- 檔案 `> 3500` 字：區段內再切成約 800 token，overlap 120
- 每個 code fence 獨立成 `chunk_type=code`，並保留 `context_prefix`

## 11. 注意事項

- `query` 與 `eval` 每次都會對查詢文字呼叫一次 embedding API。
- `ingest` 預設是增量更新；`--rebuild` 會重建文件/BM25/FAISS，但會保留 embedding 快取以降低重建成本。
- 目前預設只讀取 `source/*.md`，不遞迴子目錄。
- 若使用本地模式，請先確認 Ollama 服務已啟動且模型已下載完成。
