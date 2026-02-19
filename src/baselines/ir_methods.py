"""
Information Retrieval Baselines - BM25 and LSA retrieval methods.

Both classes index training examples and retrieve the closest match for a
given query, returning the corresponding reference output directly
(no generation involved).

Dependencies:
  - rank_bm25: pip install rank_bm25
  - scikit-learn: for TF-IDF vectorization and TruncatedSVD (LSA)
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger('baselines.ir_methods')


class BM25Retriever:
    """
    BM25 retrieval baseline.

    Indexes the training set using BM25Okapi on whitespace-tokenized inputs,
    then retrieves the output of the highest-scoring training example for each
    query.
    """

    def __init__(self):
        self._bm25 = None
        self._outputs: List[str] = []
        self._indexed = False

    def build_index(self, train_data: List[Dict]) -> None:
        """
        Build BM25 index from training data.

        Args:
            train_data: List of dicts with keys 'input' and 'output'
        """
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [item['input'].split() for item in train_data]
        self._outputs = [item['output'] for item in train_data]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._indexed = True
        logger.info(f'BM25 index built with {len(train_data)} documents')

    def retrieve(self, query: str) -> str:
        """
        Return the output of the highest-scoring training example.

        Args:
            query: Input text to match against the index

        Returns:
            Corresponding training output string
        """
        if not self._indexed:
            raise RuntimeError('BM25Retriever: call build_index() before retrieve()')

        tokenized_query = query.split()
        scores = self._bm25.get_scores(tokenized_query)
        best_idx = int(np.argmax(scores))
        return self._outputs[best_idx]

    def batch_retrieve(self, queries: List[str]) -> List[str]:
        """
        Retrieve outputs for a list of queries.

        Args:
            queries: List of input strings

        Returns:
            List of retrieved output strings (same order as queries)
        """
        results = []
        for i, q in enumerate(queries):
            results.append(self.retrieve(q))
            if (i + 1) % 50 == 0:
                logger.info(f'BM25 retrieved {i + 1}/{len(queries)}')
        return results


class LSARetriever:
    """
    LSA (Latent Semantic Analysis) retrieval baseline.

    Uses TF-IDF vectorization followed by TruncatedSVD to project inputs into a
    latent semantic space, then retrieves the nearest neighbor by cosine
    similarity.
    """

    def __init__(self, n_components: int = 100):
        """
        Args:
            n_components: Number of LSA dimensions (SVD rank)
        """
        self.n_components = n_components
        self._vectorizer = None
        self._svd = None
        self._doc_vectors: np.ndarray = None
        self._outputs: List[str] = []
        self._indexed = False

    def build_index(self, train_data: List[Dict]) -> None:
        """
        Build LSA index from training data.

        Args:
            train_data: List of dicts with keys 'input' and 'output'
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        texts = [item['input'] for item in train_data]
        self._outputs = [item['output'] for item in train_data]

        self._vectorizer = TfidfVectorizer(max_features=20000, sublinear_tf=True)
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        n_components = min(self.n_components, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = self._svd.fit_transform(tfidf_matrix)
        self._doc_vectors = normalize(reduced, norm='l2')

        self._indexed = True
        logger.info(
            f'LSA index built: {len(train_data)} docs, '
            f'{n_components} components, '
            f'explained variance={self._svd.explained_variance_ratio_.sum():.3f}'
        )

    def retrieve(self, query: str) -> str:
        """
        Retrieve the nearest-neighbor output by cosine similarity.

        Args:
            query: Input text to match

        Returns:
            Corresponding training output string
        """
        if not self._indexed:
            raise RuntimeError('LSARetriever: call build_index() before retrieve()')

        from sklearn.preprocessing import normalize

        tfidf_vec = self._vectorizer.transform([query])
        reduced = self._svd.transform(tfidf_vec)
        query_vec = normalize(reduced, norm='l2')

        similarities = self._doc_vectors @ query_vec.T
        best_idx = int(np.argmax(similarities))
        return self._outputs[best_idx]

    def batch_retrieve(self, queries: List[str]) -> List[str]:
        """
        Retrieve outputs for a list of queries.

        Args:
            queries: List of input strings

        Returns:
            List of retrieved output strings (same order as queries)
        """
        results = []
        for i, q in enumerate(queries):
            results.append(self.retrieve(q))
            if (i + 1) % 50 == 0:
                logger.info(f'LSA retrieved {i + 1}/{len(queries)}')
        return results