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
| — | — | 英文 XML 提示 + JSON 修复 + 温度0.2/seed42 | — | — | 未跑 |

## 实验决策树（拿到 report.json 后照此走）

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
4. 每项只在 val 上验证为正收益后才合并进下一次 test 提交；每天 3 个
   额度按"A 队主线实验 ×2 + 对照 ×1"分配，结果写回上表。

## 重训窗口（L2，B 队首提交之后）

`build_sft` 已改 EXTRACT_REPEAT=4（token 份额 ~70%→~80%），train.py
epochs 默认 3、checkpoint 全保留。步数 ≈ 21853/16×3 ≈ 4098，约为首训
（2024 步）的 2 倍墙钟，租卡前先排期。训完逐 checkpoint 跑 dev40：

    python3 -m scorer.infer --split train --ids-file data/dev40_ids.txt \
        --model MODEL --adapter runs/qlora-r16/checkpoint-K --output result_dev40_K.jsonl
    python3 -m scorer.evaluate result_dev40_K.jsonl --split train --ids-file data/dev40_ids.txt

选 dev40 最高分的 checkpoint，叠加决策树里已验证的开关做最终提交。

