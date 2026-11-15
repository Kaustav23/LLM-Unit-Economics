# 📊 Enterprise LLM & AI Agent Unit Economics (AI ROI) Analytics Dashboard

**Author:** Kaustav Goswami 

**Tech Stack:** Python 3.10+, Streamlit, Plotly, Pandas, NumPy

## 🎯 Executive Business Case

Most enterprises track **cloud AI token spend** in isolation without connecting it to **realized workforce productivity** or **operational failure modes** (latency-driven user abandonment and Human-in-the-Loop rework). As a result, engineering teams either:

1. **Over-provision expensive Frontier models** (`Claude Opus 4.5` at `$5.00 / $25.00` per 1M tokens) on long-context workflows where 4+ second response latencies trigger **\~28% user abandonment**, burning budget for zero business return; or

2. **Blindly down-tier complex tasks to lightweight models** (`Claude Haiku 4.5` at `$1.00 / $5.00` per 1M tokens), causing **25%–39% Human-in-the-Loop (HITL) escalation rates** where high-wage lawyers and software engineers must manually rewrite flawed AI outputs.

This **LLM & AI Agent Unit Economics (AI ROI) Terminal** bridges the gap between Cloud FinOps and CFO-level unit economics across **10,000 realistic enterprise inference logs over 90 days**, strictly parameterized by live, verifiable benchmarks.

## 📋 Phase 0 Verified Benchmarks & Primary Sources

Every pricing tier, hourly wage, latency cliff, and time-savings parameter in `data_processing.py` and `app.py` is grounded in primary external research:

| Category | Benchmark / Role | Verified Value | Publication Date | Official Source Link | 
| ----- | ----- | ----- | ----- | ----- | 
| **LLM API Pricing (Frontier)** | `Frontier-Large` (`Claude Opus 4.5`) | **\$5.00** / 1M Input | **\$25.00** / 1M Output (Cache Hit: **\$0.50**/1M) | Nov 2026 | [Anthropic Claude Platform Pricing](https://platform.claude.com/docs/en/about-claude/pricing) | 
| **LLM API Pricing (Standard)** | `Standard-Pro` (`Claude Sonnet 4.5`) | **\$3.00** / 1M Input | **\$15.00** / 1M Output (Cache Hit: **\$0.30**/1M) | Nov 2026 | [Anthropic Claude Platform Pricing](https://platform.claude.com/docs/en/about-claude/pricing) | 
| **LLM API Pricing (Fast/Lite)** | `Flash-Lite` (`Claude Haiku 4.5` / `5.5`) | **\$1.00** / 1M Input | **\$5.00** / 1M Output (`Haiku 5.5`: **\$0.10/$0.50**) | Nov 2026 | [Anthropic Claude Platform Pricing](https://platform.claude.com/docs/en/about-claude/pricing) | 
| **Cross-Provider Reference** | OpenAI `GPT-4o` / `Sol` & Google `Gemini Flash` | OpenAI Sol: **\$2.00/$10.00** | Gemini Flash: **\$0.75/$3.75** per 1M | Nov 2026 | [OpenAI API Pricing](https://developers.openai.com/api/docs/pricing) & [Google Cloud Pricing](https://cloud.google.com/model-garden/pricing) | 
| **BLS Benefits Multiplier** | Employer Costs for Employee Compensation (ECEC) | **1.30x** Multiplier (Wages = **70.0%** \[`$32.82/hr`\], Benefits = **30.0%** \[`$14.07/hr`\]) | Sept 9, 2026 | [U.S. BLS ECEC News Release](https://www.bls.gov/news.release/ecec.htm) | 
| **BLS Wage — Customer Support** | Customer Service Representatives (`SOC 43-4051`) | **\$21.53/hr** Base (`$44,770/yr`) → **\$27.99/hr** Fully Loaded (1.3x) | May 2025 OEWS (Oct 2026) | [U.S. BLS OOH: Customer Service Reps](https://www.bls.gov/ooh/office-and-administrative-support/customer-service-representatives.htm) | 
| **BLS Wage — Code Engineering** | Software Developers, QA & Testers (`SOC 15-1252`) | **\$64.44/hr** Base (`$134,040/yr`) → **\$83.77/hr** Fully Loaded (1.3x) | May 2025 OEWS (Aug 2026) | [U.S. BLS OOH: Software Developers](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm) | 
| **BLS Wage — Legal & Compliance** | Lawyers (`SOC 23-1011`) | **\$76.76/hr** Base (`$159,670/yr`) → **\$99.79/hr** Fully Loaded (1.3x) | May 2025 OEWS (Aug 2026) | [U.S. BLS OOH: Lawyers](https://www.bls.gov/ooh/legal/lawyers.htm) | 
| **BLS Wage — Marketing Ops** | Market Research Analysts & Marketing (`SOC 13-1161`) | **\$37.87/hr** Base (`$78,760/yr`) → **\$49.23/hr** Fully Loaded (1.3x) | May 2025 OEWS (Aug 2026) | [U.S. BLS OOH: Market Research Analysts](https://www.bls.gov/ooh/business-and-financial/market-research-analysts.htm) | 
| **UX Cognitive Flow Limits** | Nielsen Norman Group (NN/g) Response Limits | **0.1s** Instant | **1.0s** Uninterrupted Flow | **10.0s** Attention Limit | Jan 1993 / Updated 2014 | [NN/g: Response Times — The 3 Important Limits](https://www.nngroup.com/articles/response-times-3-important-limits/) | 
| **UX Abandonment Cliff (>3s)** | Google Research Response Time Study | **53% abandonment** when wait time exceeds **3.0–3.5 seconds** (`~3,500 ms`) | Sept 8, 2016 | [Google Blog: The Need for Mobile Speed](https://blog.google/products/admanager/the-need-for-mobile-speed/) | 
| **Productivity — Support AI** | Ticket Triage (`Customer Support`) | **+14% resolutions/hr** (\~3.4–3.8 mins saved per 25-min support workflow) | May 2025 | [Brynjolfsson, Li & Raymond (2025), NBER w31161](https://www.nber.org/papers/w31161) / [QJE](https://academic.oup.com/qje/article/140/2/889/7990658) | 
| **Productivity — Coding AI** | PR Code Review (`Code Engineering`) | **55.8% faster task completion** (\~12.8–16.8 mins saved per 28-min PR task) | Feb 13, 2023 | [Peng et al. (2023), Microsoft/GitHub Copilot Study](https://arxiv.org/abs/2302.06590) | 
| **Productivity — Legal AI** | Contract Redlining (`Legal & Compliance`) | **10%–30% time savings** (\~20% mean; \~13.5–19.2 mins saved per contract) | 2024 | [Choi, Monahan & Schwarcz (2024), Minn. Law Review](https://scholarship.law.umn.edu/minnlrev/vol109/iss1/3/) | 
| **Productivity — Marketing AI** | Campaign Copy Gen (`Marketing Ops`) | **40% time reduction** (11.0 mins saved per 27.5-min task) & **+18% quality** | July 14, 2023 | [Noy & Zhang (2023), Science / MIT News](https://news.mit.edu/2023/study-finds-chatgpt-boosts-worker-productivity-writing-0714) | 

## 🏗️ Production Architecture (OpenTelemetry → Snowflake / BigQuery → Streamlit)

While `data_processing.py` generates a self-contained 10,000-row CSV (`ai_telemetry_logs.csv`) for immediate local and Streamlit Cloud execution, in an enterprise production environment this dashboard sits on top of a modern **OpenTelemetry + Cloud Data Warehouse FinOps stack**:

1. **Instrumentation Layer (OpenTelemetry GenAI Semantic Conventions + LiteLLM / Kong AI Gateway):**

   * Every LLM/Agent API call emits an OpenTelemetry span (`gen_ai.client.operation.duration`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.request.model`) enriched with business attributes (`department`, `use_case`, `user_abandoned`, `hitl_escalated`, `csat_score`).

2. **Streaming & Storage Layer (Kafka / Kinesis → Snowflake or Google BigQuery):**

   * Raw telemetry spans land in an append-only bronze table (`RAW_AI_TELEMETRY_SPANS`).

   * dbt (Data Build Tool) transforms raw spans into a silver/gold FinOps mart (`FCT_AI_UNIT_ECONOMICS`), joining token usage with **provider rate cards** and **Workday / BLS loaded labor rate tables** to compute `api_cost_usd` and `labor_value_saved_usd` (enforcing `$0.00` value on abandoned or HITL-escalated calls).

3. **Serving & Decision Layer (Streamlit BI & Prescriptive Router):**

   * `app.py` queries the FinOps mart (or local CSV snapshot) to power executive KPIs, the **Smart Model Routing Simulator**, and the **Automated Prescriptive Action Engine**.

## 📂 Repository Structure

* `data_processing.py` — Modular data pipeline parameterized by Phase 0 benchmarks; generates and audits `ai_telemetry_logs.csv` (10,000 rows, 15 columns, zero `NaN`s).

* `ai_telemetry_logs.csv` — 10,000 grounded enterprise AI inference logs across 90 days (`2026-08-17` to `2026-11-14`).

* `app.py` — Multi-tab Streamlit executive dashboard with plain-English tooltips, interactive routing simulator, Plotly visuals, and automated prescriptive recommendations.

* `requirements.txt` — Pinned production dependencies (`streamlit`, `pandas`, `numpy`, `plotly`).

* `README.md` — Executive documentation, benchmark audit table, and production architecture guide.

## 🚀 Quickstart & Local Setup Instructions

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate and audit the 10,000-row telemetry CSV
python data_processing.py

# 3. Launch the Streamlit Executive Dashboard
streamlit run app.py

```