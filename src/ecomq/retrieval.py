"""BM25 检索（DEVLOG S26）：D5 的中文召回与难负例、D3 的改写效果评测共用。

稀疏矩阵实现，只依赖 numpy 与 scipy（CCDS 的环境里没有 scikit-learn，见 S26）；分词方式由 analyzer 决定：
- char_ngrams：字的一元 + 二元组合，用于中文短标题（不需要分词词典）；
- word_tokens：英文按字母数字切分并小写，用于 ESCI 商品标题。
"""
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


def char_ngrams(text: str) -> list[str]:
    """字的一元 + 二元组合；空白不参与。"""
    chars = [c for c in text.lower() if not c.isspace()]
    return chars + [a + b for a, b in zip(chars, chars[1:])]


class BM25:
    """稀疏矩阵实现的 BM25（k1=0.9, b=0.4，与 Anserini 默认相同）。文档侧预先算好每个词的权重，
    查询打分 = 查询中出现的词（去重）在文档里的权重之和。"""

    def __init__(self, docs: list[str], analyzer, k1: float = 0.9, b: float = 0.4):
        self.analyzer, self.vocab = analyzer, {}
        tf = self._counts(docs, fit=True)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.w_t = tf.T.tocsr()  # 词 × 文档，查询只取出现的词所在行

    def _counts(self, texts: list[str], fit: bool) -> csr_matrix:
        """词频矩阵（文本 × 词）。fit=True 时建立词表；查询里没见过的词直接忽略。"""
        indptr, indices, data = [0], [], []
        for t in texts:
            for tok, c in Counter(self.analyzer(t)).items():
                j = self.vocab.get(tok)
                if j is None:
                    if not fit:
                        continue
                    j = self.vocab[tok] = len(self.vocab)
                indices.append(j)
                data.append(c)
            indptr.append(len(indices))
        return csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int64), np.array(indptr)),
                          shape=(len(texts), len(self.vocab)))

    def topk(self, queries: list[str], k: int, chunk: int = 64) -> list[list[tuple[int, float]]]:
        q = self._counts(queries, fit=False)
        q.data[:] = 1.0  # 查询里重复的 n-gram 只算一次
        out = []
        for s in range(0, q.shape[0], chunk):
            scores = (q[s : s + chunk] @ self.w_t).toarray()
            idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
            for r in range(scores.shape[0]):
                top = idx[r][np.argsort(-scores[r, idx[r]])]
                out.append([(int(j), float(scores[r, j])) for j in top])
        return out


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
