import os
import random
import typing as t
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction


def _calc_bleu(reference: t.List[str], hypothesis: str, weight: t.Sequence[float]) -> float:
    return nltk.translate.bleu_score.sentence_bleu(
        reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1
    )


def selfbleu(
    sentences: t.List[str],
    ngram: int,
    sample_size: t.Optional[int] = None,
    n_processes: t.Optional[int] = None,
) -> float:
    """
    Compute Sel-BLEU score for a list of sentences.

    Args:
        sentences: The list of sentences to be used.
        ngram: N-gram used for Self-BLEU.
        sample_size: If set, only ``sample_size`` sentences will be randomly sampled to compute the score.
        n_processes: Use multiprocessing, can speed up computation for large sets of sentences.

    Returns:
        The Self-BLEU score.
    """
    if sample_size is not None:
        random.shuffle(sentences)
        sentences = sentences[0:sample_size]

    tokenized = []
    for text in sentences:
        text = nltk.word_tokenize(text)
        tokenized.append(text)

    weight = tuple((1.0 / ngram for _ in range(ngram)))
    sentence_num = len(tokenized)
    result = list()
    if n_processes == 1 or n_processes is None:
        for index in range(sentence_num):
            hypothesis = tokenized[index]
            other = tokenized[:index] + tokenized[index + 1 :]
            result.append(_calc_bleu(other, hypothesis, weight))
        return sum(result) / len(result)
    else:
        pool = Pool(os.cpu_count())
        for index in range(sentence_num):
            hypothesis = tokenized[index]
            other = tokenized[:index] + tokenized[index + 1 :]
            result.append(pool.apply_async(_calc_bleu, args=(other, hypothesis, weight)).get())

        score = 0.0
        cnt = 0
        for i in result:
            score += i
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
