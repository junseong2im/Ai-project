from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Using fallback vector storage.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using basic similarity search.")


logger = logging.getLogger(__name__)


@dataclass
class FragranceDocument:
    """향수 지식 문서 구조"""
    id: str
    content: str
    metadata: Dict[str, Any]
    category: str  # 'ingredient', 'composition', 'emotion', 'culture'
    embedding: Optional[np.ndarray] = None


@dataclass 
class RetrievalResult:
    """검색 결과 구조"""
    document: FragranceDocument
    similarity_score: float
    relevance_rank: int


class FragranceKnowledgeBase:
    """향수 전문 지식 베이스"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else Path("data/fragrance_knowledge")
        self.documents: List[FragranceDocument] = []
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """향수 전문 지식 로드"""
        # 기본 향료 지식
        ingredient_knowledge = {
            "베르가못": {
                "category": "ingredient",
                "content": "베르가못은 이탈리아 칼라브리아 지역의 시트러스 과일로, 상쾌하고 깔끔한 향이 특징입니다. Earl Grey 차의 향미로도 유명하며, 탑 노트로 주로 사용됩니다. 감정적으로는 활력과 긍정적 에너지를 선사합니다.",
                "metadata": {
                    "family": "citrus",
                    "volatility": "high", 
                    "emotional_effect": ["energizing", "uplifting", "fresh"],
                    "chemical_components": ["limonene", "linalool", "bergamotene"],
                    "season": ["spring", "summer"],
                    "time_of_day": ["morning", "afternoon"]
                }
            },
            "라벤더": {
                "category": "ingredient", 
                "content": "라벤더는 프로방스의 대표적인 향료로, 진정 효과가 뛰어난 허브입니다. 플로럴하면서도 깨끗한 향이 특징이며, 심신 안정에 도움을 줍니다. 미들 노트로 주로 활용됩니다.",
                "metadata": {
                    "family": "floral",
                    "volatility": "medium",
                    "emotional_effect": ["calming", "relaxing", "peaceful"],
                    "chemical_components": ["linalool", "camphor", "eucalyptol"],
                    "season": ["spring", "autumn"], 
                    "time_of_day": ["evening", "night"]
                }
            },
            "바닐라": {
                "category": "ingredient",
                "content": "바닐라는 마다가스카르산이 최고급으로 여겨지는 달콤하고 따뜻한 향료입니다. 베이스 노트로 사용되며, 편안함과 감성적 따뜻함을 제공합니다. 겨울철과 저녁 시간에 특히 적합합니다.",
                "metadata": {
                    "family": "gourmand",
                    "volatility": "low",
                    "emotional_effect": ["comforting", "warm", "sensual"],
                    "chemical_components": ["vanillin", "vanillic acid"],
                    "season": ["autumn", "winter"],
                    "time_of_day": ["evening", "night"]
                }
            }
        }
        
        # 향수 조합 지식
        composition_knowledge = {
            "시트러스_플로럴": {
                "category": "composition",
                "content": "시트러스와 플로럴의 조합은 상쾌함과 우아함을 동시에 표현하는 클래식한 구성입니다. 베르가못, 레몬과 같은 시트러스 탑 노트에 로즈, 자스민 등의 플로럴 미들 노트를 결합하여 봄날의 정원을 연상시킵니다.",
                "metadata": {
                    "style": "classic",
                    "season": ["spring", "summer"],
                    "occasion": ["daytime", "casual", "romantic"],
                    "target_emotion": ["fresh", "romantic", "elegant"]
                }
            },
            "우디_스파이시": {
                "category": "composition", 
                "content": "우디와 스파이시의 조합은 깊이 있고 남성적인 매력을 표현합니다. 시더우드, 샌달우드의 따뜻한 나무 향에 블랙 페퍼, 카다몸 등의 향신료가 더해져 성숙하고 세련된 분위기를 연출합니다.",
                "metadata": {
                    "style": "sophisticated",
                    "season": ["autumn", "winter"],
                    "occasion": ["evening", "formal", "business"],
                    "target_emotion": ["confident", "mature", "mysterious"]
                }
            }
        }
        
        # 감정-향수 매핑 지식
        emotion_knowledge = {
            "로맨틱": {
                "category": "emotion",
                "content": "로맨틱한 감정을 표현하는 향수는 주로 플로럴과 달콤한 노트를 중심으로 구성됩니다. 로즈, 피오니, 바닐라, 머스크가 핵심 소재이며, 부드럽고 감성적인 분위기를 연출합니다.",
                "metadata": {
                    "key_ingredients": ["rose", "peony", "vanilla", "musk", "jasmine"],
                    "avoid_ingredients": ["leather", "tobacco", "heavy woods"],
                    "intensity": "medium",
                    "projection": "intimate"
                }
            },
            "에너제틱": {
                "category": "emotion",
                "content": "활력과 에너지를 표현하는 향수는 시트러스와 그린 노트가 주를 이룹니다. 베르가못, 그레이프프루트, 민트, 유칼립투스 등이 상쾌한 에너지를 제공하며, 운동이나 활동적인 상황에 적합합니다.",
                "metadata": {
                    "key_ingredients": ["bergamot", "grapefruit", "mint", "eucalyptus", "ginger"],
                    "avoid_ingredients": ["heavy musks", "vanilla", "amber"],
                    "intensity": "high",
                    "projection": "strong"
                }
            }
        }
        
        # 한국 문화 특화 지식
        cultural_knowledge = {
            "한국의_계절감": {
                "category": "culture",
                "content": "한국의 사계절은 향수 선택에 중요한 영향을 미칩니다. 봄에는 벚꽃을 연상시키는 가벼운 플로럴, 여름에는 시원한 수박과 오이 향, 가을에는 단풍과 국화향, 겨울에는 따뜻한 계피와 생강 향이 선호됩니다.",
                "metadata": {
                    "spring": ["cherry_blossom", "light_floral", "green_tea"],
                    "summer": ["watermelon", "cucumber", "marine"],
                    "autumn": ["chrysanthemum", "persimmon", "woody"],
                    "winter": ["cinnamon", "ginger", "warm_spices"]
                }
            },
            "한국_전통향": {
                "category": "culture", 
                "content": "한국 전통 향료에는 침향, 백단향, 정향 등이 있으며, 이들은 명상과 정신적 평안을 위해 사용되어 왔습니다. 현대 향수에서는 이러한 전통 향료를 모던하게 해석하여 한국적 정체성을 표현합니다.",
                "metadata": {
                    "traditional_materials": ["agarwood", "sandalwood", "clove"],
                    "modern_interpretation": ["zen", "meditation", "spiritual"],
                    "cultural_context": ["temple", "traditional_house", "ceremony"]
                }
            }
        }
        
        # 모든 지식을 문서로 변환
        all_knowledge = {
            **ingredient_knowledge,
            **composition_knowledge,
            **emotion_knowledge,
            **cultural_knowledge
        }
        
        for doc_id, knowledge in all_knowledge.items():
            doc = FragranceDocument(
                id=doc_id,
                content=knowledge["content"],
                metadata=knowledge["metadata"],
                category=knowledge["category"]
            )
            self.documents.append(doc)
    
    def get_documents_by_category(self, category: str) -> List[FragranceDocument]:
        """카테고리별 문서 검색"""
        return [doc for doc in self.documents if doc.category == category]
    
    def search_documents(self, query: str, category: Optional[str] = None) -> List[FragranceDocument]:
        """키워드 기반 문서 검색"""
        results = []
        target_docs = self.get_documents_by_category(category) if category else self.documents
        
        for doc in target_docs:
            if query.lower() in doc.content.lower():
                results.append(doc)
        
        return results


class VectorRetrieval:
    """벡터 기반 검색 시스템"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        한국어 특화 임베딩 모델 초기화
        jhgan/ko-sroberta-multitask: 한국어 다중 작업 특화 모델
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 벡터 스토어 초기화
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        elif FAISS_AVAILABLE:
            self._init_faiss()
        else:
            self._init_simple_store()
            
        self.documents: Dict[str, FragranceDocument] = {}
    
    def _init_chromadb(self) -> None:
        """ChromaDB 초기화"""
        self.client = chromadb.PersistentClient(path="./chroma_fragrance_db")
        
        # 컬렉션 생성 또는 기존 컬렉션 가져오기
        try:
            self.collection = self.client.get_collection(
                name="fragrance_knowledge",
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="jhgan/ko-sroberta-multitask"
                )
            )
        except:
            self.collection = self.client.create_collection(
                name="fragrance_knowledge",
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="jhgan/ko-sroberta-multitask"
                )
            )
    
    def _init_faiss(self) -> None:
        """FAISS 초기화"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        self.doc_ids: List[str] = []
    
    def _init_simple_store(self) -> None:
        """간단한 벡터 스토어 초기화"""
        self.embeddings: List[np.ndarray] = []
        self.doc_ids: List[str] = []
    
    def add_documents(self, documents: List[FragranceDocument]) -> None:
        """문서들을 벡터 스토어에 추가"""
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_model.encode(
                    f"{doc.content} {str(doc.metadata)}",
                    convert_to_numpy=True
                )
            
            self.documents[doc.id] = doc
            
            if CHROMADB_AVAILABLE:
                self._add_to_chromadb(doc)
            elif FAISS_AVAILABLE:
                self._add_to_faiss(doc)
            else:
                self._add_to_simple_store(doc)
    
    def _add_to_chromadb(self, doc: FragranceDocument) -> None:
        """ChromaDB에 문서 추가"""
        self.collection.add(
            documents=[doc.content],
            metadatas=[doc.metadata],
            ids=[doc.id]
        )
    
    def _add_to_faiss(self, doc: FragranceDocument) -> None:
        """FAISS에 문서 추가"""
        # L2 정규화 (코사인 유사도를 위해)
        normalized_embedding = doc.embedding / np.linalg.norm(doc.embedding)
        self.index.add(normalized_embedding.reshape(1, -1))
        self.doc_ids.append(doc.id)
    
    def _add_to_simple_store(self, doc: FragranceDocument) -> None:
        """간단한 스토어에 문서 추가"""
        self.embeddings.append(doc.embedding)
        self.doc_ids.append(doc.id)
    
    def retrieve(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """쿼리에 대한 관련 문서 검색"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        if CHROMADB_AVAILABLE:
            return self._retrieve_from_chromadb(query, top_k, category_filter)
        elif FAISS_AVAILABLE:
            return self._retrieve_from_faiss(query_embedding, top_k, category_filter)
        else:
            return self._retrieve_from_simple_store(query_embedding, top_k, category_filter)
    
    def _retrieve_from_chromadb(self, query: str, top_k: int, category_filter: Optional[str]) -> List[RetrievalResult]:
        """ChromaDB에서 검색"""
        where_filter = {"category": category_filter} if category_filter else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        retrieval_results = []
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            doc = self.documents[doc_id]
            similarity = 1 - distance  # 거리를 유사도로 변환
            
            retrieval_results.append(RetrievalResult(
                document=doc,
                similarity_score=similarity,
                relevance_rank=i + 1
            ))
        
        return retrieval_results
    
    def _retrieve_from_faiss(self, query_embedding: np.ndarray, top_k: int, 
                            category_filter: Optional[str]) -> List[RetrievalResult]:
        """FAISS에서 검색"""
        # L2 정규화
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # 검색 수행
        similarities, indices = self.index.search(normalized_query.reshape(1, -1), top_k)
        
        retrieval_results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # 카테고리 필터링
            if category_filter and doc.category != category_filter:
                continue
            
            retrieval_results.append(RetrievalResult(
                document=doc,
                similarity_score=float(similarity),
                relevance_rank=i + 1
            ))
        
        return retrieval_results
    
    def _retrieve_from_simple_store(self, query_embedding: np.ndarray, top_k: int,
                                   category_filter: Optional[str]) -> List[RetrievalResult]:
        """간단한 스토어에서 검색"""
        if not self.embeddings:
            return []
        
        # 코사인 유사도 계산
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 상위 k개 결과 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieval_results = []
        rank = 1
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # 카테고리 필터링
            if category_filter and doc.category != category_filter:
                continue
            
            retrieval_results.append(RetrievalResult(
                document=doc,
                similarity_score=float(similarities[idx]),
                relevance_rank=rank
            ))
            rank += 1
        
        return retrieval_results


class FragranceRAGSystem:
    """향수 전문 RAG (Retrieval-Augmented Generation) 시스템"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base = FragranceKnowledgeBase(knowledge_base_path)
        self.vector_retrieval = VectorRetrieval()
        
        # 지식 베이스를 벡터 스토어에 인덱싱
        self._index_knowledge_base()
        
        logger.info("FragranceRAGSystem 초기화 완료")
    
    def _index_knowledge_base(self) -> None:
        """지식 베이스를 벡터 스토어에 인덱싱"""
        self.vector_retrieval.add_documents(self.knowledge_base.documents)
        logger.info(f"{len(self.knowledge_base.documents)}개 문서를 벡터 스토어에 인덱싱 완료")
    
    def retrieve_context(self, query: str, top_k: int = 3, 
                        category: Optional[str] = None) -> List[RetrievalResult]:
        """쿼리에 대한 관련 컨텍스트 검색"""
        return self.vector_retrieval.retrieve(query, top_k, category)
    
    def get_fragrance_recommendations(self, emotion: str, context: str, 
                                    season: Optional[str] = None) -> Dict[str, Any]:
        """감정과 상황에 기반한 향수 추천"""
        # 복합 쿼리 구성
        query_parts = [f"감정: {emotion}", f"상황: {context}"]
        if season:
            query_parts.append(f"계절: {season}")
        
        query = " ".join(query_parts)
        
        # 다양한 카테고리에서 검색
        ingredient_results = self.retrieve_context(query, top_k=2, category="ingredient")
        composition_results = self.retrieve_context(query, top_k=2, category="composition") 
        emotion_results = self.retrieve_context(query, top_k=1, category="emotion")
        cultural_results = self.retrieve_context(query, top_k=1, category="culture")
        
        return {
            "query": query,
            "ingredient_knowledge": [
                {
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "relevance_score": result.similarity_score
                }
                for result in ingredient_results
            ],
            "composition_knowledge": [
                {
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "relevance_score": result.similarity_score
                }
                for result in composition_results
            ],
            "emotion_knowledge": [
                {
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "relevance_score": result.similarity_score
                }
                for result in emotion_results
            ],
            "cultural_knowledge": [
                {
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "relevance_score": result.similarity_score
                }
                for result in cultural_results
            ]
        }
    
    def enhance_recipe_with_context(self, base_recipe: Dict[str, Any], 
                                   user_query: str) -> Dict[str, Any]:
        """기본 레시피를 컨텍스트 정보로 향상"""
        # 레시피의 노트들에 대한 전문 지식 검색
        all_notes = (base_recipe.get('top_notes', []) + 
                    base_recipe.get('middle_notes', []) + 
                    base_recipe.get('base_notes', []))
        
        enhanced_info = {}
        
        for note in all_notes:
            note_results = self.retrieve_context(note, top_k=1, category="ingredient")
            if note_results:
                enhanced_info[note] = {
                    "description": note_results[0].document.content,
                    "properties": note_results[0].document.metadata,
                    "relevance": note_results[0].similarity_score
                }
        
        # 조합에 대한 전문 지식
        composition_query = f"조합: {', '.join(all_notes[:3])}"  # 상위 3개 노트로 쿼리
        composition_results = self.retrieve_context(composition_query, top_k=2, category="composition")
        
        return {
            **base_recipe,
            "enhanced_note_info": enhanced_info,
            "composition_advice": [
                {
                    "advice": result.document.content,
                    "style_info": result.document.metadata,
                    "relevance": result.similarity_score
                }
                for result in composition_results
            ],
            "knowledge_confidence": self._calculate_knowledge_confidence(enhanced_info)
        }
    
    def _calculate_knowledge_confidence(self, enhanced_info: Dict[str, Any]) -> float:
        """지식 베이스 커버리지 기반 신뢰도 계산"""
        if not enhanced_info:
            return 0.0
        
        total_relevance = sum(
            info.get("relevance", 0) for info in enhanced_info.values()
        )
        avg_relevance = total_relevance / len(enhanced_info)
        
        # 커버리지와 평균 관련성을 결합한 신뢰도
        coverage = len(enhanced_info) / max(1, len(enhanced_info))  # 현재는 1.0
        confidence = (avg_relevance + coverage) / 2
        
        return min(confidence, 1.0)
    
    def add_user_knowledge(self, content: str, category: str, metadata: Dict[str, Any]) -> None:
        """사용자 정의 지식 추가"""
        doc_id = f"user_{len(self.knowledge_base.documents)}"
        
        new_doc = FragranceDocument(
            id=doc_id,
            content=content,
            category=category,
            metadata=metadata
        )
        
        self.knowledge_base.documents.append(new_doc)
        self.vector_retrieval.add_documents([new_doc])
        
        logger.info(f"사용자 지식 추가: {doc_id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        category_counts = {}
        for doc in self.knowledge_base.documents:
            category_counts[doc.category] = category_counts.get(doc.category, 0) + 1
        
        return {
            "total_documents": len(self.knowledge_base.documents),
            "category_distribution": category_counts,
            "embedding_model": "jhgan/ko-sroberta-multitask",
            "vector_store": "ChromaDB" if CHROMADB_AVAILABLE else "FAISS" if FAISS_AVAILABLE else "Simple",
            "embedding_dimension": self.vector_retrieval.embedding_dim
        }