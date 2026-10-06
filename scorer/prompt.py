"""Zero-shot instructions for the numbered-sentence protocol.

One adapter, two calls. Extraction returns ISSUE lines only. Future-argument
generation is one issue at a time so a short line cannot shift the alignment.
"""

from __future__ import annotations

from scorer.segment import Segmented, segment_sample

EXTRACT_INSTRUCTION = """你在做谈判文本的观点抽取。只输出 ISSUE 行，不要解释，不要 JSON。
每行格式：ISSUE 议题名 ||| 立场 ||| 句子编号
规则：
1. 议题名一般是 4 到 16 个字的名词短语。第一条是更宽的主议题，后面是不同侧面的子议题，不要把主议题拆字重抄。
2. 立场只能是 support、oppose、neutral，表示正文对这个议题名的态度。国际新闻报道里 neutral 最多，不要默认 support。联合声明和政策文件里 support 最多。记者会里 support 仍是多数；只有正文明确否定时才标 oppose，不要把整篇记者会标成 oppose。
3. 句子编号只能来自文中的 [Sxx]，每条议题 1 到 3 个，用英文逗号连接。证据不要复述原文。
4. 议题数量按正文实际侧面决定，常见是 4 或 5 个。不要为了凑数制造侧面，也不要漏掉明显不同的侧面。
5. 文档类型写在每篇抬头里，立场判断要跟着类型走。
"""

FUTURE_INSTRUCTION = """你在根据已经确定的议题、立场和原文证据，写一条后续观点。只输出一行 FUTURE 开头的话。多数大约 80 字；证据很短时可以更短，把机制写全时可以到 180 字，不要为了凑字数复读证据。
这条话要沿现有机制往下推，用到证据里的实词，可以写“将”“应”“未来”。不要新造文中没有的条约、日期或金额。不要复读整句证据。
"""

STANCE_INSTRUCTION = """你在复核单条议题的立场标注。只输出一行：STANCE support、STANCE oppose 或 STANCE neutral。
support：正文在推进、落实或正面评价该议题。oppose：正文明确反对、抵制或阻止该议题，只有证据句明确反对时才给。neutral：只陈述进展、分歧或中性事实，没有明确倾向。
"""


def stance_check_messages(issue: dict, doc_type: str = "") -> list[dict[str, str]]:
    chain = "\n".join(f"- {span}" for span in issue.get("argument_chain") or [])
    user = (
        STANCE_INSTRUCTION
        + f"\n文档类型：{doc_type or '未知'}\n议题：{issue.get('issue_name')}\n当前标注：{issue.get('stance')}\n证据句：\n{chain}\n"
    )
    return [
        {"role": "system", "content": "你是立场复核器。关闭思考，只输出一行 STANCE。"},
        {"role": "user", "content": user},
    ]


def extraction_messages(sample: dict, segmented: Segmented | None = None) -> list[dict[str, str]]:
    segmented = segmented or segment_sample(sample)
    user = EXTRACT_INSTRUCTION + "\n" + segmented.prompt_body
    return [
        {"role": "system", "content": "你是观点抽取器。关闭思考，只输出 ISSUE 行。"},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """The string inference feeds the model, with thinking closed.

    Training masks this prefix and learns only the assistant answer that is
    appended after it. Both sides call this function so the prefix cannot drift.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def future_messages(issue: dict) -> list[dict[str, str]]:
    chain = "\n".join(f"- {span}" for span in issue.get("argument_chain") or [])
    user = (
        FUTURE_INSTRUCTION
        + f"\n议题：{issue.get('issue_name')}\n立场：{issue.get('stance')}\n证据：\n{chain}\n"
    )
    return [
        {"role": "system", "content": "你是论点推演器。关闭思考，只输出一行 FUTURE。"},
        {"role": "user", "content": user},
    ]
