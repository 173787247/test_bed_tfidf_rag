# 增强TF-IDF RAG系统

## 项目概述

本项目实现了一个增强的TF-IDF RAG（检索增强生成）系统，集成了语义搜索、智能重排序和高级摘要生成功能。该系统专门针对中文文本优化，提供了完整的文档检索和问答解决方案。

## 核心功能

### 1. 增强TF-IDF检索
- **传统TF-IDF**: 基于词频-逆文档频率的文本检索
- **语义搜索**: 使用LSA（潜在语义分析）进行语义相似度计算
- **混合检索**: 结合TF-IDF和语义搜索，提供更准确的检索结果

### 2. 智能重排序
- **标题匹配**: 优先显示标题中包含查询词的文档
- **内容密度**: 根据查询词在文档中的出现频率调整排序
- **长度惩罚**: 避免过长文档占据优势
- **多因子评分**: 综合考虑多个因素进行智能排序

### 3. 高级摘要生成
- **智能摘要**: 自动生成查询结果的智能摘要
- **相关度筛选**: 优先展示高相关度的文档
- **结构化输出**: 提供清晰的文档列表和摘要信息

### 4. 中文优化
- **中文分词**: 使用jieba进行中文分词
- **停用词过滤**: 内置中文停用词列表
- **文本预处理**: 针对中文文本的特殊处理

## 快速开始

### 基本使用

`python
from src.search_engine.rag_tab.enhanced_tfidf_rag import EnhancedTfidfRAG

# 初始化服务
rag_service = EnhancedTfidfRAG()

# 添加文档
documents = [
    {
        'id': 'doc_001',
        'title': '人工智能发展历史',
        'content': '人工智能（AI）是计算机科学的一个分支...',
        'summary': 'AI发展历史概述',
        'metadata': {'category': 'technology', 'year': 2023}
    }
]
rag_service.add_documents(documents)

# 构建索引
rag_service.build_index()

# 执行查询
result = rag_service.rag_query("什么是人工智能？", top_k=5)
print(result['summary'])
`

## 技术架构

### 核心组件

- **EnhancedTfidfRAG**: 主要的RAG服务类
- **文档管理**: add_documents(), build_index()
- **检索功能**: search(), rerank_results(), rag_query()
- **摘要生成**: generate_summary()
- **模型管理**: save_model(), load_model(), get_stats()

### 技术栈

- **scikit-learn**: TF-IDF向量化和LSA降维
- **jieba**: 中文分词
- **numpy**: 数值计算
- **pandas**: 数据处理
- **pickle**: 模型序列化

## 安装依赖

`ash
pip install scikit-learn jieba numpy pandas
`

## 使用场景

### 1. 文档检索系统
- 企业内部知识库搜索
- 学术论文检索
- 技术文档查询

### 2. 问答系统
- 基于文档的问答
- 智能客服系统
- 知识图谱查询

### 3. 内容推荐
- 相关文档推荐
- 内容相似度计算
- 个性化内容推送

## 性能优化

### 1. 缓存机制
- 查询结果缓存，避免重复计算
- 可配置缓存大小，平衡内存使用和性能

### 2. 索引优化
- 使用稀疏矩阵存储TF-IDF向量
- LSA降维减少计算复杂度
- 批量处理文档，提高索引构建效率

### 3. 内存管理
- 支持大型文档集合
- 可配置的特征数量和组件数
- 模型序列化，支持持久化存储

## 许可证

本项目采用MIT许可证。

## 联系方式

- 项目主页: https://github.com/173787247/test_bed_tfidf_rag
- 问题反馈: 请在GitHub Issues中提交
- 技术交流: 欢迎提交Pull Request
