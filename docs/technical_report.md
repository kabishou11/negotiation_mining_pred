# 复现技术文档（初赛）

> 对照《06_规则与FAQ》§6 复现材料要求逐项撰写。标注【服务器待填】的
> 项在复现环境就绪后补齐（见 `SUBMISSIONS.md` 复现材料清单）。
> 本文档由两队共用（方法各自成节）；两队提交材料中各自附本文件全文。

## 1. 总体方案

两支独立队伍参加同一赛题，代码零共享（`teamA_end2end/scripts/audit_no_sharing.py`
对两侧全部字符串字面量做机械审计，框架强制词汇白名单化）：

- **队伍一**：QLoRA 微调 Qwen3-32B + 两段式推理（观点抽取 / 逐议题推演）
  + 确定性后处理。中文行协议（`ISSUE … ||| … ||| [Sxx]`）。
- **队伍二**：同一基座**零训练**、端到端单次调用直出 JSON。英文提示 +
  XML 包裹正文，不做分句编号，证据直接从原文拷贝。

## 2. 指定基座

- 模型：Qwen3-32B，官方开源发布版；除该基座外不使用任何其他生成模型。
- 来源 / commit hash：【服务器待填：权重目录版本信息与 config.json
  校验值，写入 `repro/base_model.txt`】
- Tokenizer 校验值：【服务器待填】
- 加载方式：`transformers.AutoModelForCausalLM`，bitsandbytes 4bit NF4
  （double quant，compute dtype fp16/bf16 按卡支持），`trust_remote_code=True`。

## 3. 数据处理流程（队伍一）

- 分句：quote-aware 规则分句，主切分 `。！？` 与换行；中文引号内不切；
  >160 字子句按 `；;` 切分；短句（<32 字）回并（≤180 字）；超 220 字按
  逗号硬切。常量经参数扫描验证（`scripts/sweep_segment.py`，含裁剪后
  金标保留率终局指标）。`[Sxx]` 编号跨文档按 publish_date 排序。
- SFT 目标：金标证据 → 句 ID 对齐（最小 1-3 连续句覆盖，实测覆盖
  99.71%）；抽取记录 ×4（含 oppose 的文档 ×8）、future 记录 ×1，
  共 21,213 条；同名议题按队列对齐（复赛主议题×3 预留）。
- 留出：dev40 分层验证集（40 文档，≥8 含 oppose），训练只用 2,360 份。
- **外部数据：无。伪标签：无。**（规则允许，本方案未使用。）

## 4. 训练方法（队伍一）

- QLoRA：NF4 + double quant；r=16, alpha=32, dropout=0.05，
  目标模块 q/k/v/o/gate/up/down。
- 序列 6144（按显存自动选择），超长样本跳过不截断（实测最长 5,005 token）。
- epochs 3、lr 1e-4 cosine、warmup 0.1、weight decay 0.01、有效批 16
  （1×梯度累积）、paged_adamw_8bit、neftune_noise_alpha 5.0、seed 0。
- 损失仅打在 assistant 答案与结束符上（推理前缀全掩码；掩码不对齐则
  拒绝加载权重）。
- checkpoint 每 200 步全保留（约 20 个，~20GB），按 dev40 分数选优，
  不以步数定。

## 5. 推理流程

### 队伍一
- **贪心解码**（`do_sample=False`，思考模式关闭）——同硬件同版本下
  结果完全确定。
- 可选第二遍：`--stance-check`（逐议题立场复核，严格解析、失败保留
  原值）、`--min-issues N`（少于 N 个议题时补抽一次，仅在更多议题时
  采纳）。
- 采样变体（`--temperature 0.7 --top-p 0.9 --seed 0`，逐样本播种）仅
  用于同基座融合实验（见 §6），不改变默认提交路径。

### 队伍二
- 温度 0.2 / top_p 0.9 / seed 42，逐样本 seed = 42 + 样本编号。
- JSON 修复循环：代码围栏剥离、首层括号配平、尾逗号清理、截断补括号
  （`_close_json` 两遍扫描），最多重试 2 次。
- few-shot 1 例：来自官方训练集（确定性选取，含立场多样性准则），
  文档截断到句边界且仅保留证据全部落在所示文本内的议题。

## 6. 后处理规则（全部披露；`scorer/postprocess.py`、`scorer/merge_results.py`）

- **证据裁剪**：>64 字证据裁至 ~55 字子句窗（词法启发式或
  `--semantic-trim` 的 bge 语义选窗），结果恒为原文子串。
- **证据重排 / MMR**：候选池 = 模型句 ± `--pool-size` 邻句（0=全句），
  按 bge(议题名, 句) 余弦选择；`--mmr λ` 折扣与已选句的冗余
  （金标 2-3 条证据覆盖不同侧面，去冗余）。
- **checkpoint 融合**：多份结果按议题名 bigram 相似度聚类，优先文件
  （最优 checkpoint）锚定——其议题全部保留；非优先文件单票议题过滤；
  立场计票、平票归优先文件；证据按出现频次并集（近重复 ≥0.8 归并）。
  依据：规则允许同一基座不同检查点/推理结果融合。
- **议题补抽 / 立场复核**：见 §5。
- **future 点题**：future 未触及议题名 2-gram 时前置“关于{议题}，”
  （训练集实测 97.9% 金标 future 点题）。

## 7. 辅助模型与工具（披露）

| 资源 | 用途 | 备注 |
|---|---|---|
| BAAI/bge-small-zh-v1.5 | 证据重排 / 语义裁剪（本地评分同款嵌入） | 离线缓存 |
| bert-base-chinese（bert-score 库） | 本地评分复刻（非提交路径） | 离线缓存 |
| Python 标准库（difflib/re/json/scipy） | 子串回填、解析、二部图匹配 | 无外部服务 |

不使用 jieba 等其他分词器、不调用任何在线 API 或远程推理接口。

## 8. 运行环境

- 依赖清单：【服务器待填：`pip freeze > repro/environment.txt`】
- 硬件：单卡 48GB（训练与推理）；CUDA/驱动版本【服务器待填】
- 随机种子：队伍一 seed 0（贪心=确定性）；队伍二 seed 42 + 逐样本偏移。
- 资源预估：训练 ~40GB 显存、约 3,978 步；推理 4bit ~20GB、
  300 样本约 2 小时（含逐议题 future 生成）。

## 9. 标准运行命令

- 队伍一：`repro/train.sh`（训练）、`repro/run.sh`（推理）、
  `repro/val.sh`（验证诊断）、`repro/preflight.sh`（环境自检）。
- 队伍二：`teamA_end2end/repro/run.sh`。
- 提交前校验：`python3 -m scorer.check_submit result.jsonl --split test`。
- 输入输出路径与完整流程见 `SUBMISSIONS.md` runbook。

## 10. 非确定性说明

- 队伍一：贪心解码 + 固定种子 + 确定性后处理 → 同环境输出逐字节一致。
- 队伍二：温度采样已逐样本播种；但断点续跑会改变调用顺序，输出可能
  漂移——提交以单次完整运行为准。
- 融合与全部后处理均为确定性算法（计票平票规则已固定）。
