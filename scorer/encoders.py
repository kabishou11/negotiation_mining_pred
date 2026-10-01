"""Encoders for the published judge, plus an offline exact-match double.

`OrthogonalEncoder` maps each distinct string to its own axis. Identical text
has cosine 1 and everything else has cosine 0, which is what the self-check
uses to prove the arithmetic. It is not a substitute for bge-small-zh-v1.5.

`BgeEncoder` is the official embedding: CLS pooling of
`BAAI/bge-small-zh-v1.5`, then L2 normalisation, which is the pooling this
checkpoint was trained with. `BertScore` is bert-base-chinese F1 with
`rescale_with_baseline=False` so the published α stays inside [0, 1].
Neither is imported until called.
"""

from __future__ import annotations

import hashlib


class OrthogonalEncoder:
    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if text not in self._index:
                self._index[text] = len(self._index)
        dim = max(1, len(self._index))
        vecs: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            vec[self._index[text]] = 1.0
            vecs.append(vec)
        return vecs


class HashingEncoder:
    """Bag of character bigrams. Order-insensitive, available without torch.

    Useful as a smoke test that *similar* strings outrank unrelated ones.
    Leaderboard estimates still need `BgeEncoder`.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            vec = [0.0] * self.dim
            if not text:
                vecs.append(vec)
                continue
            grams = [text] if len(text) == 1 else [text[i : i + 2] for i in range(len(text) - 1)]
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign
            vecs.append(vec)
        return vecs


class BgeEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        torch = self.torch
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        return hidden.cpu().tolist()


class BertScore:
    """Call form expected by `score_sample`: `alpha_fn(pred_issue, gold_issue)`."""

    def __init__(self, model_type: str = "bert-base-chinese", num_layers: int | None = None) -> None:
        self.model_type = model_type
        self.num_layers = num_layers
        self._score = None

    def _fn(self):
        if self._score is None:
            from bert_score import score as bert_score

            self._score = bert_score
        return self._score

    def __call__(self, pred_issue: dict, gold_issue: dict) -> float:
        def as_text(issue: dict) -> str:
            name = str(issue.get("issue_name") or "")
            chain = "\n".join(issue.get("argument_chain") or [])
            if name:
                return name + "\n" + chain
            return chain

        score = self._fn()
        kwargs = {
            "model_type": self.model_type,
            "lang": "zh",
            "rescale_with_baseline": False,
            "verbose": False,
        }
        if self.num_layers is not None:
            kwargs["num_layers"] = self.num_layers
        _p, _r, f1 = score([as_text(pred_issue)], [as_text(gold_issue)], **kwargs)
        return float(f1[0])
