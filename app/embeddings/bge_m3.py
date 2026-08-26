from __future__ import annotations


"""Локальный инстанс bge-m3 через FlagEmbedding."""

from .base import BaseEmbedder, EmbeddingResult


class BgeM3Embedder(BaseEmbedder):
    """bge-m3 через FlagEmbedding: dense и sparse из одного прохода."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        batch_size: int = 16,
        max_length: int = 1024,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel

        self._max_length = max_length
        self._model = BGEM3FlagModel(
            model_name, use_fp16=use_fp16, devices=device)

    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        output = self._model.encode(
            texts,
            batch_size=len(texts),
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        sparse = output["lexical_weights"]

        return [
            EmbeddingResult(
                dense=[float(x) for x in dense[i]],
                sparse={
                    str(token): float(weight)
                    for token, weight in sparse[i].items()
                    if float(weight) > 0
                },
            )
            for i in range(len(texts))
        ]
