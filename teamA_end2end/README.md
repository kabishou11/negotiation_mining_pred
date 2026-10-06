# A 队——端到端方案（单 adapter 直出 JSON）

> 文件夹：`teamA_end2end/`。严格隔离：禁止复制 B 队任何内容，详见 `../DIVERGENCE_CONTRACT.md`。
> 基座：Qwen3-32B（唯一生成模型）。资料依据：`../../赛题三_整理md/`。

## 0. 当前状态（2026-10-06 实装）

**已实装为零训练推理流水线**，交给第二支队伍账号提交（第一支队伍账号跑
`../scorer/` 的 QLoRA 流水线）。与原§5 计划的差异：

- **不训练**：直接用 base Qwen3-32B 零样本，`src/infer_e2e.py` 单文件，无 adapter。
  原因：算力全部给了第一队，而本队只需要一条独立、合规、互查不判相似的第二成绩线。
- `configs/train_e2e.yaml` 与 `src/train_e2e.py` 未创建；若后续要给本队单独训练，
  再按§5 草案补。`src/infer_e2e.py` 与 `--adapter` 兼容所需改动很小，暂不做。
- 本目录代码 **零引用 `../scorer/`**（数据文件只读共用 `data/prelim/`；
  本地评分用 `python3 -m scorer.evaluate` 属测量工具，不属于提交管线）。

实现要点（与第一队刻意的不同）：英文系统提示＋XML 包裹正文（`prompts/extract_en.xml`）、
**不做分句编号**（证据直接从 `<document>` 原文拷贝）、单次调用直出完整 JSON、
JSON 修复循环（围栏剥离＋括号配平＋尾逗号，最多重试 2 次）、
证据子串强制：非子串时用 difflib 最长公共块回填（≥10 字且覆盖率 ≥1/2），否则丢弃、
温度 0.2／top_p 0.9／seed 42（每样本 seed=42+编号，断点续跑稳定）、
兜底与第一队不同名不同规则（“核心议题”＋最长句 vs “文本主议题”＋首句）。

一键复现：

```sh
./repro/run.sh /path/to/Qwen3-32B result.jsonl test
python3 -m scorer.check_submit result.jsonl --split test   # 验收
```

## 1. 路线（冻结送审 Grok）

`full_text（＋文档类型／发布日期）` → **一次** Qwen3-32B 调用 → 完整 `{"issue_list","future_argument"}`。
不设检索模块。格式靠 JSON-mode＋修复循环保证。

## 2. 目录规划（文件尚未创建，仅占位命名）

```text
teamA_end2end/
├── README.md（本文件）
├── configs/train_e2e.yaml      # 草案见 §5，非最终
├── prompts/system_en.xml.md    # 草案见 §4，英文＋XML 风格
├── src/                        # 空，Grok 通过后才写 train_e2e／infer_e2e／validate_e2e
├── scripts/                    # 空，后续放 run_train.sh／run_infer.sh（一键复现）
└── repro/                      # 空，后续放 requirements.txt／环境锁定／随机种子
```

## 3. 数据约定

- 输入：train／val／test.jsonl（初赛 `docs[0].full_text` 单文档）。内部：`{正文，元信息}`，`party_id=null`，`mode=prelim`。
- 输出：每样本 `issue_list` 通常 4–5 个＋等长同序 `future_argument`。`stance∈{support,oppose,neutral}`。
- 证据规则：每条 `argument_chain` 元素必须是 `full_text` 的原文子串（失败则回退最长公共子串句）。禁止改写（Q7 会扣分）。
- 复赛开关：`mode=semifinal` → 主议题展开×3（`party_id` 为 P1／P2／P3），子议题只给发声方。

## 4. 提示词草案（英文＋XML，A 队专用风格）

```text
SYSTEM: You are an analyst for negotiation opinion mining. Output VALID JSON ONLY.
RULES: (1) issue_list[0] is the main issue. (2) stance in {support,oppose,neutral}.
(3) Each argument_chain element MUST be an exact substring of <document>.
(4) len(future_argument)==len(issue_list), same order.
INPUT:
<document type="{doc_type}" date="{publish_date}">{full_text}</document>
OUTPUT SCHEMA: {"issue_list":[{"issue_name":str,"stance":str,"argument_chain":[str]}],"future_argument":[str]}
```

## 5. 训练／推理配置草案（按 V100 约束）

```yaml
# configs/train_e2e.yaml——草案，待 Grok 点评
base_model: Qwen3-32B  # 官方 commit hash 待定，写进 repro
peft: {method: qlora, quant: nf4, r: 64, alpha: 16, targets: [q,k,v,o], dropout: 0.05}
seq_len: 4096          # 超长截断；先在 Qwen3-8B 上调通代理流程
train: {epochs: 2, batch: 1, grad_accum: 16, lr: 2e-4, ckpt: true, offload: cpu}
decode: {temperature: 0.2, top_p: 0.9, seed: 42, max_new_tokens: 2048}
validator: {enum_check: true, len_equality: true, substring_check: true, issue_prior: [4,5]}
```

- 推理栈：`transformers＋accelerate＋bnb-4bit，device_map=auto` 跑 2×V100。坏 JSON 最多重试 3 次。
- 本地评分：用 `bge-small-zh-v1.5`（匹配）＋`bert-base-chinese`（修正系数）独立实现一套，与 B 队代码不共享。

## 6. 里程碑／风险

- 10/04 零样本基线提交探底分。10/07 做 50 样本过拟合必须通过。10/10 冻结。
- 最大风险：V100 显存爆／太慢（无 BF16）。对策：小模型代理联调，最终 adapter 视情况租 A100-80G 一次训成；坚持序列 4096，绝不做全量微调。
