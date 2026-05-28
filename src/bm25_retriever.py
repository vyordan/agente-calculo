"""
Módulo de recuperación con BM25.
"""
import re
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None

    def index(self, chunks: list):
        self.corpus = chunks
        self.tokenized_corpus = [self._tokenize(doc) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, question: str, k: int = 5):
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(question)
        # Asegura que la query no esté vacía
        if not tokenized_query:
            return []

        # scores es un numpy array 1D
        scores = self.bm25.get_scores(tokenized_query)

        # Si no hay scores (corpus vacío), salir
        if len(scores) == 0:
            return []

        # Obtener índices ordenados por score descendente
        # Convertir el array a lista para evitar posibles ambigüedades
        scores_list = scores.tolist()
        top_k_indices = sorted(
            range(len(scores_list)),
            key=lambda i: scores_list[i],
            reverse=True
        )[:k]

        max_score = max(scores_list) if scores_list else 1.0
        results = []
        for idx in top_k_indices:
            # Normalizar a float Python estándar
            normalized = float(scores_list[idx] / max_score) if max_score > 0 else 0.0
            results.append({
                'content': self.corpus[idx],
                'score': normalized
            })
        return results

    def _tokenize(self, text: str):
        """Tokenización simple: minúsculas, limpia puntuación, split."""
        text = text.lower()
        text = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', text)
        return text.split()