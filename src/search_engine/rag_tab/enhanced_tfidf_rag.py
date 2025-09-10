"""
Enhanced TF-IDF RAG Service
增强的TF-IDF RAG服务，集成语义搜索和智能重排序功能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import jieba
import re
from datetime import datetime
import json
import pickle
import os

class EnhancedTfidfRAG:
    def __init__(self, max_features: int = 10000, n_components: int = 100, min_df: int = 2, max_df: float = 0.95, stop_words: Optional[List[str]] = None):
        self.max_features = max_features
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words or self._load_chinese_stopwords()
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words=self.stop_words, tokenizer=self._chinese_tokenizer, ngram_range=(1, 2))
        self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.documents = []
        self.document_ids = []
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.is_trained = False
        self.query_cache = {}
        self.cache_size = 1000
    
    def _load_chinese_stopwords(self) -> List[str]:
        return ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
    
    def _chinese_tokenizer(self, text: str) -> List[str]:
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        words = jieba.lcut(text)
        words = [word for word in words if len(word) > 1 and word not in self.stop_words]
        return words
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        for doc in documents:
            if 'id' not in doc or 'content' not in doc:
                continue
            self.document_ids.append(doc['id'])
            self.documents.append({'id': doc['id'], 'content': doc['content'], 'metadata': doc.get('metadata', {}), 'title': doc.get('title', ''), 'summary': doc.get('summary', '')})
    
    def build_index(self) -> None:
        if not self.documents:
            raise ValueError("没有文档可以建立索引")
        print(f"开始构建索引，文档数量: {len(self.documents)}")
        doc_texts = []
        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc['content']} {doc.get('summary', '')}"
            doc_texts.append(full_text)
        print("构建TF-IDF矩阵...")
        self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
        print("构建LSA语义矩阵...")
        self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
        self.is_trained = True
        print(f"索引构建完成！TF-IDF矩阵形状: {self.tfidf_matrix.shape}, LSA矩阵形状: {self.lsa_matrix.shape}")
    
    def search(self, query: str, top_k: int = 10, use_semantic: bool = True, alpha: float = 0.7) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        if not self.is_trained:
            raise ValueError("索引尚未构建，请先调用build_index()")
        cache_key = f"{query}_{top_k}_{use_semantic}_{alpha}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        query_processed = self._chinese_tokenizer(query)
        query_text = ' '.join(query_processed)
        if not query_text.strip():
            return []
        query_tfidf = self.vectorizer.transform([query_text])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        scores = tfidf_similarities.copy()
        if use_semantic:
            query_lsa = self.lsa_model.transform(query_tfidf)
            lsa_similarities = cosine_similarity(query_lsa, self.lsa_matrix).flatten()
            scores = alpha * tfidf_similarities + (1 - alpha) * lsa_similarities
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append((doc['id'], float(scores[idx]), doc.get('summary', doc['content'][:200] + '...'), doc.get('metadata', {})))
        if len(self.query_cache) < self.cache_size:
            self.query_cache[cache_key] = results
        return results
    
    def rag_query(self, query: str, top_k: int = 10, use_semantic: bool = True, rerank: bool = True, generate_summary: bool = True) -> Dict[str, Any]:
        start_time = datetime.now()
        results = self.search(query, top_k, use_semantic)
        if rerank and results:
            results = self.rerank_results(query, results)
        summary = ""
        if generate_summary:
            summary = self.generate_summary(query, results)
        response = {'query': query, 'results': results, 'summary': summary, 'total_results': len(results), 'processing_time': (datetime.now() - start_time).total_seconds(), 'search_config': {'use_semantic': use_semantic, 'rerank': rerank, 'top_k': top_k}}
        return response
    
    def rerank_results(self, query: str, results: List[Tuple[str, float, str, Dict[str, Any]]], rerank_factor: float = 0.3) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        if not results:
            return results
        query_words = set(self._chinese_tokenizer(query.lower()))
        def calculate_rerank_score(item):
            doc_id, original_score, summary, metadata = item
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if not doc:
                return original_score
            title = doc.get('title', '').lower()
            content = doc['content'].lower()
            title_matches = sum(1 for word in query_words if word in title)
            title_boost = title_matches * 0.1
            content_matches = sum(1 for word in query_words if word in content)
            content_density = content_matches / max(len(query_words), 1)
            density_boost = content_density * 0.05
            length_penalty = min(len(content) / 1000, 1) * 0.02
            rerank_score = original_score + title_boost + density_boost - length_penalty
            return rerank_score
        reranked_results = []
        for item in results:
            new_score = calculate_rerank_score(item)
            doc_id, _, summary, metadata = item
            reranked_results.append((doc_id, new_score, summary, metadata))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results
    
    def generate_summary(self, query: str, results: List[Tuple[str, float, str, Dict[str, Any]]], max_length: int = 500) -> str:
        if not results:
            return "未找到相关文档。"
        high_relevance = [r for r in results if r[1] > 0.3]
        if not high_relevance:
            high_relevance = results[:3]
        summary_parts = []
        summary_parts.append(f"针对查询「{query}」找到 {len(results)} 个相关文档：\n")
        for i, (doc_id, score, summary, metadata) in enumerate(high_relevance[:5], 1):
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                title = doc.get('title', f'文档{doc_id}')
                summary_parts.append(f"{i}. {title} (相关度: {score:.3f})")
                summary_parts.append(f"   {summary[:100]}...")
                summary_parts.append("")
        full_summary = "\n".join(summary_parts)
        if len(full_summary) > max_length:
            full_summary = full_summary[:max_length] + "..."
        return full_summary
    
    def get_stats(self) -> Dict[str, Any]:
        return {'document_count': len(self.documents), 'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_trained else 0, 'tfidf_matrix_shape': self.tfidf_matrix.shape if self.is_trained else (0, 0), 'lsa_matrix_shape': self.lsa_matrix.shape if self.is_trained else (0, 0), 'is_trained': self.is_trained, 'cache_size': len(self.query_cache)}

def create_sample_documents() -> List[Dict[str, Any]]:
    return [{'id': 'doc_001', 'title': '人工智能发展历史', 'content': '人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI的发展可以追溯到20世纪50年代，当时科学家开始探索机器学习和神经网络的概念。', 'summary': 'AI发展历史概述', 'metadata': {'category': 'technology', 'year': 2023}}, {'id': 'doc_002', 'title': '机器学习算法原理', 'content': '机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习模式。常见的机器学习算法包括线性回归、决策树、随机森林、支持向量机等。深度学习是机器学习的一个子领域，使用多层神经网络。', 'summary': '机器学习算法介绍', 'metadata': {'category': 'technology', 'year': 2023}}, {'id': 'doc_003', 'title': '自然语言处理技术', 'content': '自然语言处理（NLP）是人工智能的重要应用领域，专注于让计算机理解和生成人类语言。NLP技术包括文本分类、情感分析、机器翻译、问答系统等。近年来，大型语言模型如GPT、BERT等取得了突破性进展。', 'summary': 'NLP技术概述', 'metadata': {'category': 'technology', 'year': 2023}}, {'id': 'doc_004', 'title': '计算机视觉应用', 'content': '计算机视觉是人工智能的另一个重要分支，致力于让计算机理解和分析视觉信息。应用包括图像识别、目标检测、人脸识别、医学影像分析等。卷积神经网络（CNN）是计算机视觉的核心技术。', 'summary': '计算机视觉技术介绍', 'metadata': {'category': 'technology', 'year': 2023}}, {'id': 'doc_005', 'title': '深度学习框架', 'content': '深度学习框架为开发者提供了构建和训练神经网络的工具。流行的框架包括TensorFlow、PyTorch、Keras等。这些框架提供了自动微分、GPU加速、模型部署等功能，大大降低了深度学习的门槛。', 'summary': '深度学习框架介绍', 'metadata': {'category': 'technology', 'year': 2023}}]

if __name__ == "__main__":
    print("=== 增强TF-IDF RAG服务演示 ===")
    rag_service = EnhancedTfidfRAG()
    sample_docs = create_sample_documents()
    rag_service.add_documents(sample_docs)
    rag_service.build_index()
    queries = ["什么是人工智能？", "机器学习有哪些算法？", "自然语言处理的应用", "计算机视觉技术", "深度学习框架"]
    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        result = rag_service.rag_query(query, top_k=3)
        print(f"找到 {result['total_results']} 个相关文档")
        print(f"处理时间: {result['processing_time']:.3f}秒")
        print(f"\n摘要:\n{result['summary']}")
        print("\n详细结果:")
        for i, (doc_id, score, summary, metadata) in enumerate(result['results'], 1):
            print(f"{i}. 文档ID: {doc_id}, 相关度: {score:.3f}")
            print(f"   摘要: {summary}")
            print(f"   元数据: {metadata}")
            print()
    print("\n=== 模型统计信息 ===")
    stats = rag_service.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
