# B 队——两阶段方案（先抽取后预测＋检索）——仅规划，未写代码

> 文件夹：`teamB_twostage/`。严格隔离：禁止复制 A 队任何内容，详见 `../DIVERGENCE_CONTRACT.md`。
> 基座：Qwen3-32B（唯一生成模型）。辅助：开源分词器＋`bge-m3` 召回（须申报、可离线）。

## 1. 路线（冻结送审 Grok）

- **第一阶段** `文本 → issue_list`（adapter-1）：先分句（jieba）＋`bge-m3` Top-K 召回，再让大模型精选 2–3 个原文原句＋判立场（`oppose` 用高精度阈值）。
- **第二阶段** `逐议题 → future_argument[i]`（adapter-2）：输入为 `(议题名，立场，证据)`，逐条生成后按第一阶段顺序拼接。
- 三个独立程序（`extract／predict／merge`）对 A 队单体结构——结构上天然不同。

## 2. 目录规划（文件尚未创建，仅占位命名）

```text
teamB_twostage/
├── README.md（本文件）
├── configs/stage1.yaml         # 草案见 §5
├── configs/stage2.yaml         # 草案见 §5
├── prompts/stage1_zh.md        # 草案见 §4，中文＋Markdown
├── prompts/stage2_zh.md        # 草案见 §4
├── src/                        # 空，Grok 通过后才写 extract／predict／merge
├── scripts/                    # 空，后续放 run_stage1.sh／run_stage2.sh
└── repro/                      # 空，依赖版本与 A 队错开锁定／随机种子独立
```

## 3. 数据约定

- 第一阶段输入：`full_text` → 分句（jieba，与 A 队任何分词器都不同）。每拟议题召回 Top-K=12，再由大模型过滤到 2–3 句。
- 第一阶段输出：`party_id=null`（初赛）的 `issue_list`。第二阶段输出：同序拼接的 `future_argument`。
- 复赛开关：`party_id` 语义与 A 队相同但代码另写；主议题×3 在 `merge` 里展开，子议题按召回命中过滤（无命中则不建三元组）。

## 4. 提示词草案（中文＋Markdown，B 队专用风格）

第一阶段：

```text
## 任务：从谈判长文本抽取主议题与子议题
## 要求：1.首个为主议题 2.立场仅support/oppose/neutral 3.论据必须是原文原句
## 输入文档（类型/日期/正文分段）：
### 文档类型：{doc_type} ### 日期：{publish_date} ### 候选句Top-K：{recall_sents}
### 全文：{full_text}
## 输出：{"issue_list":[{"issue_name":"...","stance":"...","argument_chain":["原句1","原句2"]}]}
```

第二阶段：

```text
## 任务：基于议题+立场+证据，推演该议题后续可能观点（一段话，不编造新事实）
## 输入：议题={issue_name} 立场={stance} 证据={argument_chain}
## 输出：{"future_argument": "..."}
```

## 5. 训练／推理配置草案

```yaml
# stage1.yaml——草案
peft: {method: qlora, quant: nf4, r: 32, alpha: 16, targets: [q,v], dropout: 0.05}
seq_len: 3072
recall: {model: bge-m3, top_k: 12, splitter: jieba}
stance_recalib: {oppose_threshold: 0.7, confirm_head: true}

# stage2.yaml——草案
peft: {method: qlora, quant: nf4, r: 32, alpha: 16, targets: [q,v]}
seq_len: 2048
decode: {temperature: 0.5, top_p: 0.95, seed: 1234}
```

- 推理：优先 `AutoAWQ INT4`；V100 跑不动则降级 `HF 4bit 卸载`。种子／依赖库与 A 队不同。
- 本地评分复刻与 A 队公式相同、代码独立重写一份。

## 6. 里程碑／风险

- 10/04 纯检索基线先提交一次。10/07 第一阶段召回率在验证集≥80%，否则调 Top-K。10/10 冻结两个 adapter。
- 最大风险：误差传递（第一阶段漏议题→第二阶段按 N=max 直接得 0）。对策：议题数先验 5＋合并时长度修复；`oppose` 召回优先保。
