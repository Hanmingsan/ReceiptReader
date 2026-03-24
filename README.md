# ReceiptReader

A personal project for automated receipt processing and expense tracking. The goal is to build a lightweight, on-device model that can extract structured financial data from receipt images — running locally on Apple Silicon without relying on external APIs.

---

# ReceiptReader（收據讀取器）

一個自動化處理收據與記錄支出的個人專案。目標是打造一個輕量的本地端模型，能夠從收據圖片中提取結構化財務資料，並在 Apple Silicon 上離線運行，無需依賴外部 API。

---

## What I'm Building / 正在開發的內容

The pipeline runs in two stages:

```
Receipt Images
      │
      ▼
 EasyOCR (text extraction)
      │
      ▼
 Google Gemini (structured labelling)    ◄── internal tooling only
      │
      ▼
 JSONL Dataset  ──►  Fine-tune Qwen3.5-0.8b  ──►  On-device inference
```

**Stage 1 — Dataset Construction:** Each receipt image is passed through EasyOCR to extract raw text. The image and OCR result are then sent to Google Gemini, which returns a structured JSON object containing the merchant name, date/time, currency, amount, and expense category. Results are validated with Pydantic and accumulated into a JSONL dataset. Gemini is used purely as a labelling tool during development — it will not be part of the final product.

**Stage 2 — Model Fine-tuning *(in progress)*:** The labelled dataset is used to fine-tune `Qwen/Qwen3.5-0.8b-Base`, a small open-source language model. The aim is a model compact enough to run on-device via Apple Silicon MPS, replacing Gemini for inference entirely.

---

**兩個階段的流程：**

**第一階段 — 資料集建構：** 每張收據圖片先經由 EasyOCR 提取原始文字，再將圖片與 OCR 結果傳送給 Google Gemini，由其回傳包含商家名稱、日期時間、貨幣、金額及消費類別的結構化 JSON 物件。結果透過 Pydantic 驗證後累積為 JSONL 資料集。Gemini 在開發階段僅作為標記工具使用，不會出現在最終產品中。

**第二階段 — 模型微調（進行中）：** 使用標記資料集對 `Qwen/Qwen3.5-0.8b-Base` 進行微調，目標是打造一個足夠輕巧、可透過 Apple Silicon MPS 在本地運行的模型，完全取代 Gemini 進行推理。

---

## Tech Stack / 技術棧

- **OCR:** EasyOCR — supports Traditional Chinese and English / 支援繁體中文與英文
- **Labelling:** Google Gemini (development only) / Google Gemini（僅用於開發階段）
- **Validation:** Pydantic
- **Model:** `Qwen/Qwen3.5-0.8b-Base` via Hugging Face Transformers
- **Training:** Hugging Face `trl` + `accelerate`
- **Target device:** Apple Silicon (MPS) / 目標裝置：Apple Silicon（MPS）
- **Package manager:** `uv`, Python 3.13

---

## Supported Categories & Currencies / 支援的類別與貨幣

**Expense Categories / 消費類別**

| Category | 類別 |
|---|---|
| Dining | 餐飲 |
| Transportation | 交通 |
| Groceries | 超市／雜貨 |
| Shopping | 購物 |
| Services | 服務 |
| Accommodation | 住宿 |
| Entertainment | 娛樂 |
| Others | 其他 |

**Currencies / 貨幣**

| Code | Currency | Decimal Places |
|---|---|---|
| HKD | Hong Kong Dollar 港幣 | 2 |
| USD | US Dollar 美元 | 2 |
| EUR | Euro 歐元 | 2 |
| GBP | British Pound 英鎊 | 2 |
| CNY | Chinese Yuan 人民幣 | 2 |
| JPY | Japanese Yen 日圓 | 0 |
| TWD | Taiwan Dollar 新台幣 | 0 |

---

## Current Status / 目前進度

| Component | Status |
|---|---|
| OCR module | ✅ Complete |
| Gemini labelling pipeline | ✅ Complete |
| Pydantic data validation | ✅ Complete |
| Dataset generation | ✅ Complete |
| Tokenisation for training | ✅ Complete |
| Train/test split | 🚧 In progress |
| Training loop | 🚧 In progress |
| On-device inference | 📋 Planned |

---

## License / 授權

MIT
