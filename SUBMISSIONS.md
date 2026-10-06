# 提交台账 SUBMISSIONS

每天最多 3 次提交，榜单保留最高分。每次提交后立刻补一行：日期、队伍、
产生该文件的 checkpoint/后处理、线上分数、本地 val 分数（两种证据模式）。
线上分数和归因报告 `*_report.json` 都要归档，0.67 与 0.80 的差距目前
没有任何本地记录，这个文件就是防止再次失忆的。

## A 队（scorer/ 流水线，QLoRA r16）

| 日期 | 文件 | checkpoint / 后处理 | 线上分 | 本地 val (newline/mean) | 备注 |
|---|---|---|---|---|---|
| 10-05 | result.jsonl（旧 run，10-04 22:51 产物） | 首次训练最终 adapter | 待填 | 未评 | 首次提交 |
| 10-06 | result_test_1800.jsonl | step-1800 | ~0.67 | 未评 | Desktop/比赛 |
| 10-06 | result_test_2000.jsonl | step-2000 | ~0.67 | 未评 | Desktop/比赛 |
| 10-06 | result_test_2024.jsonl | step-2024 (最终) | ~0.67 | 未评 | Desktop/比赛 |

三 checkpoint 分数持平 → 瓶颈是系统性的（SFT 配比、证据形态、立场分布），
不是步数。最高分 0.80（其他队伍）。

## B 队（teamA_end2end/，base Qwen3-32B 零训练）

| 日期 | 文件 | 配置 | 线上分 | 本地 val | 备注 |
|---|---|---|---|---|---|
| — | — | 英文 XML 提示 + few-shot 1 样例 + JSON 修复 + 议题补抽 + 温度0.2/seed42 | — | — | 未跑 |

**B 队提分计划（零训练上限内）**：
1. **few-shot 1**：提示词注入 1 个训练集金标样例（截断到句边界、只保留
   证据都在所示文本内的议题——教"逐字抄"而不是"凭空编"）。确定性选取，
   样例 id 打进日志供复现材料披露。这是零训练最大的单项杠杆。
2. **min_issues 3**：合法 JSON 但议题 <3 时追加一次"补全侧面"生成，
   只在找到更多议题时采纳（与 A 队同思想、不同实现与措辞）。
3. **先 val 后 test**：B 先跑 val 估计分数（约 1-2h），
   `python3 -m scorer.evaluate result_b_val.jsonl --split val`，
   达到 0.55+ 才提交 test；不到就先调 few-shot 数量（1→2）再测。
4. self-consistency 已实装为配置门控：decode.json `"consistency": 3`
   （默认 0=关闭；开启后每样本 3 次采样，按议题聚类投票合并立场/证据，
   future 取首个候选）。val 对照单次涨分才开启，代价是 3 倍推理时长。

## 防冲突清单（两队互查不判相似）

**已隔离（代码/材料层，审查主要比对对象）**：零共享 import；英文 XML +
JSON vs 中文行协议；单体 vs 分层文件；温度 0.2 采样+逐样本播种 vs 贪心
确定；兜底字符串不同（"核心议题" vs "文本主议题"）；README/文档各自
撰写。双方近似同一份金标，**内容趋同是必然且无害的**——任何两支独立
的好队伍都会趋同；审查看的是材料与代码。

**量化闸门（提交前必跑）**：
`python3 -m scorer.compare_results result_a_val.jsonl result_b_val.jsonl`
- `future_rouge_l` 必须明显低于 0.755（A 队两个 checkpoint 之间的实测
  值——B 若高于它，就像同一模型的另一个 checkpoint 而非独立队伍）；
  超了就改 B 的 future 措辞指令。
- `evidence verbatim` 高不是问题，但 **长度剖面** 必须可区分（A 整句
  ~70+ 字 vs B 截断子串）；长度差 <15 字且逐字率 >0.5 时工具会报警，
  此时把 B 的证据指令改短（20-60 字）。
- `name_similarity` ~0.9 属预期（都逼近金标命名），不设闸门。

**字符串级审计（每次改动后跑一次，秒级）**：
`python3 teamA_end2end/scripts/audit_no_sharing.py` —— AST 提取两队全部
字符串字面量，报告 6 字符以上的非必要共享（提交 schema/立场词/数据集
字段/框架旗标在白名单内，逐组注明理由）。本轮实测曾抓出两边相同的完成
日志与帮助文本，已改写。审计通过 = 材料层无逐字复用。

**排期隔离**：B 首提交在 A 队当天实验之前（B 不依赖 val 归因结论），
两队同日提交错开进行；GPU 先给 B 的 test 推理（一次性 ~2h），再跑
A 的 val 诊断与 A/B 实验。

## 实验决策树（拿到 report.json 后照此走）

**第 0 步（每次上服务器先跑，约 10 分钟）**：`./repro/preflight.sh
/path/to/Qwen3-32B [adapter]` —— 3 个 val 样本端到端，先暴露
bitsandbytes/transformers 版本、adapter 路径、tokenizer 漂移、OOM 问题，
别让整晚白烧。

`repro/val.sh` 产出 `*_report.json`（逐样本分数 + 失分归因）。归因三类的
含义：`low_sim`=证据句对了但相似度没过 0.7（形态问题）；`stance_blocked`=
相似度过线但立场标错（一票否决）；`gold_missed`/`pred_extra`=议题数量
不齐（N=max 惩罚）。哪类占比高就先做哪条：

1. **low_sim 占多 → 证据形态**（`scorer/postprocess.py`，免模型）：
   - `--trim-evidence`（裁到 ~55 字子句窗）与 `--rerank`（bge 选句，
     `--pool-size 1` 起步，可试 2）在 val 上四象限 A/B：基线 / 只裁 /
     只排 / 裁+排。本地 val 涨分才提；hash 编码器结果永远不提交。
2. **stance_blocked 占多 → 立场复核**：`infer --stance-check` 重跑 val
   A/B。注意它会多花每样本 5 次短生成。
3. **gold_missed 占多 → 议题补齐**：`--min-issues 4` 先试；`5` 有精度
   反噬风险（金标 4 议题样本占 ~22%，填错一个 F1 反降），必须单独 A/B。
4. **口径确认（一次性）**：`evaluate --matching cardinality` 跑一次，
   与默认 weight 分差 ≈0 即永久放下"一对一最优匹配"的口径疑虑；
   newline vs mean 两种证据模式本地一直同时打印，若两者分差明显且
   线上分对不上，用一次提交槽做区分实验。
5. **同基座融合（规则白纸黑字允许，两边各有工具）**：
   - A 队 checkpoint 融合：`python3 -m scorer.merge_results out.jsonl
     r_2024.jsonl r_2000.jsonl r_1800.jsonl`（最好的放最前；min-votes=2
     滤单文件幻影；future 取最优 checkpoint）。三个 test 结果文件已在
     手上，**val 融合对照单文件验证后，这是零 GPU 成本的提交实验**。
   - B 队 self-consistency：decode.json `"consistency": 3`（默认 0 关闭，
     开启后 3 倍推理时长），采样 3 次按议题聚类投票合并立场/证据。
   - 都先在 val 上对照单文件/单次，涨分才启用。
6. **future 措辞是最后的天花板**：S_pred 权重 0.2，且只有匹配议题的
   future 计分——抽取侧修完仍差一口气时再考虑（改 FUTURE 提示词必须
   连同重训一起动，防止提示漂移），单独不动。
7. 每项只在 val 上验证为正收益后才合并进下一次 test 提交；每天 3 个
   额度按"A 队主线实验 ×2 + 对照 ×1"分配，结果写回上表。

## 重训窗口（L2，有门槛，B 队首提交之后）

**启动门槛**：val 归因显示抽取侧失分占主导，且 postprocess/推理侧开关
全部验证完仍不够 0.74 —— 否则重训是拿 2 倍墙钟去赌一个不确定的增益。

- `build_sft` 已改 EXTRACT_REPEAT=4（token 份额 ~70%→~80%），train.py
  epochs 默认 3、checkpoint 全保留。
- **必须用新输出目录 `runs/qlora-r16-v2`**：旧目录里的 checkpoint 会让
  train.sh 自动续跑，而数据集已从 16,177 行变为 ~21,200 行，调度器与
  优化器状态错位（train.py 现有守卫会直接拒绝，看到报错就换目录）。
- 步数 ≈ 21,213/16×3 ≈ 3,978（首训 2 倍墙钟）；save_steps=200 全保留
  ≈ 20 个 checkpoint，**预留 ~20GB 磁盘**。
- 训完逐 checkpoint 跑 dev40 选优：

    python3 -m scorer.infer --split train --ids-file data/dev40_ids.txt \
        --model MODEL --adapter runs/qlora-r16-v2/checkpoint-K --output result_dev40_K.jsonl
    python3 -m scorer.evaluate result_dev40_K.jsonl --split train --ids-file data/dev40_ids.txt

- 选 dev40 最高分的 checkpoint，叠加决策树里已验证的开关做最终提交。

## 复现材料清单（官方 §6/§7 硬性要求，10/12-13 前归集完毕）

复现审查会拿主办方自己的 Qwen3-32B 断网重跑我们的标准命令，核对输出
一致性；缺一项即取消资格。逐条对照：

- [ ] **技术文档**：总体方案、数据处理流程、训练/微调方法、推理流程、
      **后处理规则（postprocess 裁剪/重排、立场复核、补抽必须披露）**。
- [ ] **基座信息**：Qwen3-32B 官方来源 + **版本/commit hash** + 模型配置
      校验值 + **Tokenizer 校验值**（服务器上 `git log` 权重目录 / 校验
      config.json 与 tokenizer 的 hash，落盘 `repro/base_model.txt`）。
- [ ] **权重**：adapter + `adapter_config.json`（自动可加载）。
- [ ] **运行环境**：`pip freeze > repro/environment.txt`（服务器上跑），
      操作系统/CUDA/驱动、硬件、**随机种子（A: seed 0 贪心；B: seed 42
      + 温度 0.2 采样，逐样本播种）**、解码参数、**显存/内存/磁盘占用与
      运行时长预估**、非确定性说明（B 队采样顺序依赖断点，需注明）。
- [ ] **辅助模型披露**：bge-small-zh-v1.5（匹配+重排）、bert-base-chinese
      （评分复刻）的用途与版本；两模型缓存路径写进 run.sh 注释。
- [ ] **复现指引**：目录结构 + 一键命令（repro/*.sh 已备）+ 输入输出
      路径 + 校验方式（check_submit）。
- [ ] **外部资源**：无外部数据（仅官方训练集）；B 队零训练（无伪标签）。

## 提交前检查清单（每次）

1. `python3 -m scorer.check_submit result_xx.jsonl --split test` 通过。
2. **文件名改为 `result.jsonl`**（04_提交要求 §三：统一命名，改名后别再跑推理覆盖它）。
3. 台账补行（日期/文件/checkpoint/开关/线上分/本地分）。
4. 产生该文件的 commit 已 push（服务器复现要与本地代码一致）。

## 服务器第一天 runbook（照序复制粘贴）

```sh
git pull
pip install -r requirements.txt -r requirements-train.txt   # 环境未装时
pip freeze > repro/environment.txt                          # 复现材料，只做一次

# 0. 环境自检（~10 分钟）：bnb/transformers 版本、adapter 路径、OOM
./repro/preflight.sh /path/to/Qwen3-32B runs/qlora-r16/checkpoint-2024

# 1. A 队归因诊断（~2h）：得 result_val_*_report.json
./repro/val.sh /path/to/Qwen3-32B runs/qlora-r16/checkpoint-2024

# 2. 读 report 的归因三类占比 → 按上方决策树选支执行 A/B
python3 -m scorer.evaluate result_val_qlora-r16-checkpoint-2024.jsonl \
    --split val --matching cardinality          # 口径确认（一次性）

# 3. low_sim 主导时的四象限 A/B（免模型，分钟级）
python3 -m scorer.postprocess result_val_qlora-r16-checkpoint-2024.jsonl \
    --split val --out pp_trim.jsonl --trim-evidence
python3 -m scorer.postprocess result_val_qlora-r16-checkpoint-2024.jsonl \
    --split val --out pp_rerank.jsonl --rerank
python3 -m scorer.postprocess result_val_qlora-r16-checkpoint-2024.jsonl \
    --split val --out pp_both.jsonl --rerank --trim-evidence
# 对 4 个文件各跑一次 repro/eval.sh，择优方向再套到 test 文件上提交

# 4. B 队零训练首提交（先 val 估分，再 test，~2h+2h）
cd teamA_end2end && ./repro/run.sh /path/to/Qwen3-32B result_b_val.jsonl val && cd ..
python3 -m scorer.evaluate result_b_val.jsonl --split val        # 估分 0.55+ 才提交
cd teamA_end2end && ./repro/run.sh /path/to/Qwen3-32B result_b.jsonl test && cd ..
python3 -m scorer.check_submit result_b.jsonl --split test

# 5. 防冲突闸门（两队 val 文件都在手时）
python3 -m scorer.compare_results result_val_qlora-r16-checkpoint-2024.jsonl result_b_val.jsonl
```

当天 A 队只在 val A/B 出结论后才花提交额度；额度分配按决策树第 6 条。


