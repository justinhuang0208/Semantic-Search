# Semantic Search (OpenRouter / Ollama Embedding)

本專案提供本機優先的語意搜尋系統，目標資料是卡片式 Markdown。

- Embedding provider: OpenRouter or local Ollama
- Default model (OpenRouter): `google/gemini-embedding-001`
- Default model (Ollama): `qwen3-embedding:0.6b`
- Retrieval: `search` (BM25), `vsearch` (vector), `query` (hybrid: `BM25 + Vector + RRF`)
- Storage: SQLite + FAISS / numpy fallback
- Collection registry: collection:// scopes + context scopes
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

若想啟用 FAISS 後端，可改用：

```bash
pip install -e ".[faiss]"
```

## 3. 設定 API Key（OpenRouter 模式）

```bash
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_API_KEY>"
```

若使用本地 Ollama 模式（`--use-local-embedding`），不需要設定 `OPENROUTER_API_KEY`。

## 4. 設定 collection registry

先建立 collection，之後 `ingest` 就直接以 collection 為主：

```bash
semsearch collection add notes "/你的/Markdown/資料夾"
```

每個 collection 會自動綁定一組索引檔路徑，之後 `ingest/query/eval --collection <name>` 會優先使用這組路徑，不需要每次手動補 `--db-path` 與 `--faiss-path`。

若要使用不同的 registry 檔，可設定：

```bash
export SEMSEARCH_COLLECTIONS="data_index/collections.yml"
```

`SEMSEARCH_SOURCE` 與 `--source` 仍保留給舊流程相容使用，但新流程不建議再直接依賴它們。

## 5. 建立索引

```bash
semsearch ingest --collection notes --rebuild
```

輸出內容包含：文件數、chunk 數量、embedding 維度、SQLite 路徑與 FAISS 路徑。

若 collection 已經存在，之後可以直接用它來做增量更新：

```bash
semsearch ingest --collection notes --rebuild
```

增量更新（推薦日常使用）：

```bash
semsearch ingest --collection notes
```

本地 Ollama 模式建立索引：

```bash
ollama pull qwen3-embedding:0.6b
ollama serve
semsearch ingest --collection notes --use-local-embedding --rebuild
```

- 不帶 `--rebuild` 時，系統會比較檔案內容 hash，只更新新增/變更檔案，並刪除索引中已不存在的檔案。
- 系統會使用 embedding 快取（`content_hash + model`）避免重複呼叫 API。
- 即使使用 `--rebuild`，只要內容 hash 曾經快取過，仍會直接重用，不會再次上傳到 embedding API。

### Collection 與索引路徑的關係

- `collection` 決定「這次 ingest 要讀哪個 collection」。
- 每個 collection 可以綁定自己的 `db_path` 與 `faiss_path`。
- `ingest/query/eval --collection notes` 會自動使用 collection 綁定的索引路徑。
- 若有特殊需求，仍可在 `collection add` 或 `ingest/query/eval` 時手動覆蓋 `--db-path` 與 `--faiss-path`。
- collection registry 另外由 `--collections-path` 管理，預設會在 `data_index/collections.yml`。

範例：建立兩套獨立索引（A 與 B）

```bash
semsearch collection add A "/path/A"
semsearch collection add B "/path/B"
semsearch ingest --collection A
semsearch ingest --collection B
```

## 6. 查詢

三種模式的用途如下：

- `search`：快速關鍵字 full-text 搜尋，只走 BM25
- `vsearch`：語意相似度搜尋，只走 vector
- `query`：最佳品質 hybrid 搜尋，沿用目前 `BM25 + Vector + RRF`

```bash
semsearch search "authentication flow" --top-k 8
semsearch vsearch "how to login" --top-k 8 --use-local-embedding
semsearch query "user authentication" --top-k 8
semsearch query "MULH sign extension 高位錯誤" --top-k 8 --show-chunk-type
semsearch query "non-blocking assignment 在跨週期傳遞的重點" --collection notes --top-k 8
```

本地 Ollama 模式查詢：

```bash
semsearch vsearch "how to login" --use-local-embedding --top-k 8
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
- `ingest/vsearch/query/eval` 需要使用與索引建立時一致的 provider/model。
- 若設定不一致，系統會報錯並提示先用正確參數重新 `ingest`。

## 9. 專案結構

- `semsearch/cli.py`: CLI 入口（`ingest`, `search`, `vsearch`, `query`, `eval`）
- `semsearch/pipeline.py`: ingest/search/vsearch/query/eval 主流程
- `semsearch/markdown_ingest.py`: Markdown 解析與切塊
- `semsearch/embeddings.py`: OpenRouter/Ollama embedding client + provider resolver
- `semsearch/storage.py`: SQLite schema 與資料存取
- `semsearch/vector_index.py`: FAISS 建索引與 numpy fallback 查詢
- `semsearch/retrieval.py`: BM25、RRF、結果去重
- `semsearch/collections.py`: collection registry 與 context 管理
- `tests/golden_queries.yaml`: 第一版 golden set

## 10. Collection 與 Context

```bash
semsearch collection add notes "1 - Cards"
semsearch collection list
semsearch context add collection://notes "這個 collection 是硬體筆記"
semsearch context add collection://notes/api "API 相關內容要優先看規格"
semsearch context list notes
semsearch context rm collection://notes/api
```

- `collection` 用來定義 collection 的根目錄、掃描 mask、以及預設索引路徑。
- `context` 用來加上 collection-wide 或 path-specific 的額外背景文字。
- `search` 預設只看 `include-by-default=true` 的 collections；`vsearch` 和 `query` 也同樣支援 `--collection` 指定單一 collection。
- `search` 不需要 embedding model 或 `OPENROUTER_API_KEY`。
- `vsearch` 和 `query` 需要使用與索引建立時一致的 provider/model。

## 11. Chunk 策略

- 檔案 `< 1200` 字：整篇一塊
- 檔案 `1200 ~ 3500` 字：按 `##` 區段切
- 檔案 `> 3500` 字：區段內再切成約 800 token，overlap 120
- 每個 code fence 獨立成 `chunk_type=code`，並保留 `context_prefix`
- `#tag` 會保留在 metadata，不會進入 chunk 內容或 embedding 索引

## 12. 注意事項

- `search` 不會對查詢文字呼叫 embedding API。
- `vsearch`、`query` 與 `eval` 每次都會對查詢文字呼叫一次 embedding API。
- `ingest` 預設是增量更新；`--rebuild` 會重建文件/BM25/FAISS，但會保留 embedding 快取以降低重建成本。
- 目前預設只讀取 `source/*.md`，不遞迴子目錄。
- 若使用本地模式，請先確認 Ollama 服務已啟動且模型已下載完成。
