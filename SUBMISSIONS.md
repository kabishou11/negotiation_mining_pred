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

## 待用提交槽的实验队列（A 队）

1. newline vs mean 证据模式（本地 val 择优后提交一次确认）。
2. 证据子句裁剪（postprocess --trim-evidence，val 涨分才提）。
3. bge 证据重排（postprocess --rerank，val 涨分才提）。
4. 立场二遍校准 + 短输出补抽（infer --stance-check --min-issues 4）。
5. 重训 adapter（EXTRACT_REPEAT=4，抽取配比反转）后的全套叠加。
