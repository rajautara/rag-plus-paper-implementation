"""
RAG+ Core Implementation
Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning

This module provides the core implementation of RAG+ system as described in the research paper.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge in the RAG+ system"""
    CONCEPTUAL = "conceptual"  # Definitions, theories, descriptions
    PROCEDURAL = "procedural"  # Methods, algorithms, step-by-step processes


@dataclass
class KnowledgeItem:
    """Represents a knowledge item in the corpus"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class ApplicationExample:
    """Represents an application example demonstrating knowledge usage"""
    id: str
    knowledge_id: str
    content: str
    question: str
    answer: str
    application_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class KnowledgeApplicationPair:
    """Represents a paired knowledge item and its application"""
    knowledge: KnowledgeItem
    applications: List[ApplicationExample]
    relevance_score: float = 1.0


class RAGPlusConfig:
    """Configuration for RAG+ system"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_type: str = "openai",  # "openai" or "sentence_transformer"
        retrieval_top_k: int = 3,
        application_top_k: int = 2,
        llm_model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        domains: List[str] = None,
        cache_embeddings: bool = True,
        batch_size: int = 32,
        openai_api_key: str = None
    ):
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.retrieval_top_k = retrieval_top_k
        self.application_top_k = application_top_k
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.domains = domains or ["mathematics", "legal", "medical"]
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.openai_api_key = openai_api_key


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding implementation"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            import openai
            self.client = openai.OpenAI()
            self.model_name = model_name
            # Set dimensions based on OpenAI model
            self.dimension = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }.get(model_name, 1536)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts"""
        import openai
        
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        import openai
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence transformer implementation for embeddings"""
    
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class VectorStore(ABC):
    """Abstract base class for vector storage"""
    
    @abstractmethod
    def add_items(self, items: List[KnowledgeItem]):
        """Add items to vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[KnowledgeItem, float]]:
        """Search for similar items"""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS implementation for vector storage"""
    
    def __init__(self, dimension: int = None):
        try:
            import faiss
            self.dimension = dimension
            self.index = None
            self.items = []
            self._normalize_vectors = True  # For IP similarity, normalize vectors
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")
    
    def _initialize_index(self, dimension: int):
        """Initialize FAISS index with given dimension"""
        if self.index is None:
            self.dimension = dimension
            import faiss
            self.index = faiss.IndexFlatIP(dimension)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize vector for IP similarity"""
        if self._normalize_vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
        return vector
    
    def add_items(self, items: List[KnowledgeItem]):
        """Add items to FAISS index"""
        vectors_to_add = []
        items_to_add = []
        
        for item in items:
            if item.embedding is not None:
                # Initialize index if needed
                if self.index is None:
                    self._initialize_index(len(item.embedding))
                
                # Normalize vector for IP similarity
                normalized_embedding = self._normalize_vector(item.embedding)
                vectors_to_add.append(normalized_embedding)
                items_to_add.append(item)
        
        if vectors_to_add:
            vectors_array = np.array(vectors_to_add).astype('float32')
            self.index.add(vectors_array)
            self.items.extend(items_to_add)
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[KnowledgeItem, float]]:
        """Search for similar items"""
        if len(self.items) == 0 or self.index is None:
            return []
        
        # Normalize query vector for IP similarity
        normalized_query = self._normalize_vector(query_embedding)
        query_embedding = normalized_query.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.items)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.items[idx], float(score)))
        
        return results


class ApplicationMatcher:
    """Matches applications to knowledge items with category alignment"""
    
    def __init__(self, config: RAGPlusConfig, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.category_similarity_threshold = 0.7
    
    def match_applications_to_knowledge(
        self,
        knowledge_items: List[KnowledgeItem],
        applications: List[ApplicationExample]
    ) -> Dict[str, List[str]]:
        """
        Match applications to knowledge items using category alignment
        Returns a dictionary mapping knowledge_id to list of application_ids
        """
        knowledge_to_apps = {}
        
        # Group by domain and knowledge type for category alignment
        domain_groups = {}
        for knowledge in knowledge_items:
            domain_key = f"{knowledge.domain}_{knowledge.knowledge_type.value}"
            if domain_key not in domain_groups:
                domain_groups[domain_key] = []
            domain_groups[domain_key].append(knowledge)
        
        # Group applications by domain
        app_domain_groups = {}
        for app in applications:
            # Get the knowledge item for this app to determine domain
            knowledge_item = next((k for k in knowledge_items if k.id == app.knowledge_id), None)
            if knowledge_item:
                domain_key = f"{knowledge_item.domain}_{knowledge_item.knowledge_type.value}"
                if domain_key not in app_domain_groups:
                    app_domain_groups[domain_key] = []
                app_domain_groups[domain_key].append(app)
        
        # Match within categories
        for domain_key, knowledge_group in domain_groups.items():
            if domain_key not in app_domain_groups:
                continue
                
            app_group = app_domain_groups[domain_key]
            
            for knowledge in knowledge_group:
                knowledge_to_apps[knowledge.id] = []
                
                # Calculate semantic similarity with applications
                knowledge_embedding = knowledge.embedding
                if knowledge_embedding is None:
                    knowledge_embedding = self.embedding_model.encode_single(knowledge.content)
                
                for app in app_group:
                    # Skip if app is already linked to this knowledge
                    if app.knowledge_id == knowledge.id:
                        knowledge_to_apps[knowledge.id].append(app.id)
                        continue
                    
                    # Calculate semantic similarity
                    app_embedding = app.embedding
                    if app_embedding is None:
                        app_embedding = self.embedding_model.encode_single(app.content)
                    
                    similarity = np.dot(knowledge_embedding, app_embedding) / (
                        np.linalg.norm(knowledge_embedding) * np.linalg.norm(app_embedding)
                    )
                    
                    # Add to matches if above threshold
                    if similarity >= self.category_similarity_threshold:
                        knowledge_to_apps[knowledge.id].append(app.id)
        
        return knowledge_to_apps
    
    def manual_refine_matches(
        self,
        knowledge_to_apps: Dict[str, List[str]],
        knowledge_items: Dict[str, KnowledgeItem],
        applications: Dict[str, ApplicationExample]
    ) -> Dict[str, List[str]]:
        """
        Manual refinement step for matches
        In practice, this would involve human review
        Here we implement heuristic-based refinement
        """
        refined_matches = {}
        
        for knowledge_id, app_ids in knowledge_to_apps.items():
            knowledge = knowledge_items.get(knowledge_id)
            if not knowledge:
                continue
                
            refined_matches[knowledge_id] = []
            
            for app_id in app_ids:
                app = applications.get(app_id)
                if not app:
                    continue
                
                # Heuristic: check if application is actually relevant
                # This would be manual review in the paper
                if self._is_relevant_match(knowledge, app):
                    refined_matches[knowledge_id].append(app_id)
        
        return refined_matches
    
    def _is_relevant_match(self, knowledge: KnowledgeItem, app: ApplicationExample) -> bool:
        """Check if application is relevant to knowledge (heuristic for manual review)"""
        # Simple heuristic based on content overlap
        knowledge_words = set(knowledge.content.lower().split())
        app_words = set(app.content.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(knowledge_words.intersection(app_words))
        union = len(knowledge_words.union(app_words))
        
        if union == 0:
            return False
            
        jaccard_similarity = intersection / union
        
        # Consider relevant if similarity is above threshold
        return jaccard_similarity >= 0.1


class ApplicationGenerator:
    """Generates application examples for knowledge items"""
    
    def __init__(self, llm_client, config: RAGPlusConfig):
        self.llm_client = llm_client
        self.config = config
    
    def generate_application(self, knowledge: KnowledgeItem) -> ApplicationExample:
        """Generate application example for a knowledge item"""
        if knowledge.knowledge_type == KnowledgeType.CONCEPTUAL:
            return self._generate_conceptual_application(knowledge)
        else:
            return self._generate_procedural_application(knowledge)
    
    def _generate_conceptual_application(self, knowledge: KnowledgeItem) -> ApplicationExample:
        """Generate application for conceptual knowledge"""
        prompt = f"""
        Generate a multiple-choice question that tests understanding of this concept:
        
        Knowledge: {knowledge.content}
        
        Create a question with 4 options (A, B, C, D) where one is correct.
        The question should demonstrate practical understanding of the concept.
        
        Format:
        Question: [Your question here]
        A) [Option A]
        B) [Option B] 
        C) [Option C]
        D) [Option D]
        Correct Answer: [Correct option]
        """
        
        response = self._call_llm(prompt)
        parsed = self._parse_application_response(response, knowledge.id)
        return parsed
    
    def _generate_procedural_application(self, knowledge: KnowledgeItem) -> ApplicationExample:
        """Generate application for procedural knowledge"""
        prompt = f"""
        Create a worked example showing how to apply this procedure:
        
        Knowledge: {knowledge.content}
        
        Create a step-by-step example that demonstrates the application of this procedure.
        Include a specific problem and show how the procedure solves it.
        
        Format:
        Question: [A specific problem]
        Solution: [Step-by-step solution using the procedure]
        Answer: [Final answer]
        """
        
        response = self._call_llm(prompt)
        parsed = self._parse_application_response(response, knowledge.id)
        return parsed
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        # This would integrate with actual LLM API
        # For now, return mock response
        return "Mock LLM response"
    
    def _parse_application_response(self, response: str, knowledge_id: str) -> ApplicationExample:
        """Parse LLM response into ApplicationExample"""
        # Implement parsing logic based on response format
        return ApplicationExample(
            id=f"app_{knowledge_id}_{hash(response)}",
            knowledge_id=knowledge_id,
            content=response,
            question="Extracted question",
            answer="Extracted answer",
            application_type="generated"
        )
    
    def generate_applications_for_missing_pairs(
        self,
        knowledge_items: List[KnowledgeItem],
        existing_pairs: Dict[str, List[str]]
    ) -> List[ApplicationExample]:
        """
        Generate applications for knowledge items that don't have enough applications
        This implements the "backfill" mechanism mentioned in the paper
        """
        generated_apps = []
        
        for knowledge in knowledge_items:
            existing_app_count = len(existing_pairs.get(knowledge.id, []))
            
            # Generate if we have fewer than the desired number of applications
            if existing_app_count < self.config.application_top_k:
                apps_to_generate = self.config.application_top_k - existing_app_count
                
                for _ in range(apps_to_generate):
                    app = self.generate_application(knowledge)
                    generated_apps.append(app)
        
        return generated_apps


class Reranker:
    """Reranks retrieved knowledge-application pairs using cross-encoder"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            logger.warning("CrossEncoder not available. Install with: pip install sentence-transformers")
            self.model = None
    
    def rerank_pairs(
        self,
        query: str,
        pairs: List[KnowledgeApplicationPair],
        top_k: int = None
    ) -> List[KnowledgeApplicationPair]:
        """
        Rerank knowledge-application pairs based on query relevance
        """
        if self.model is None:
            # Fallback to original ordering
            return pairs[:top_k] if top_k else pairs
        
        # Prepare inputs for cross-encoder
        inputs = []
        for pair in pairs:
            # Combine knowledge and applications for reranking
            knowledge_text = pair.knowledge.content
            app_texts = " ".join([app.content for app in pair.applications])
            combined_text = f"{knowledge_text} {app_texts}"
            inputs.append([query, combined_text])
        
        # Get scores from cross-encoder
        scores = self.model.predict(inputs)
        
        # Sort by scores
        scored_pairs = list(zip(pairs, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        reranked_pairs = [pair for pair, _ in scored_pairs]
        return reranked_pairs[:top_k] if top_k else reranked_pairs


class KnowledgeCorpus:
    """Manages the knowledge corpus"""
    
    def __init__(self, config: RAGPlusConfig):
        self.config = config
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        
        # Initialize embedding model based on type
        if config.embedding_type == "openai":
            self.embedding_model = OpenAIEmbedding(config.embedding_model)
        else:
            self.embedding_model = SentenceTransformerEmbedding(config.embedding_model)
        
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.get_dimension())
    
    def add_knowledge(self, knowledge: KnowledgeItem):
        """Add knowledge item to corpus"""
        if knowledge.embedding is None:
            knowledge.embedding = self.embedding_model.encode_single(knowledge.content)
        
        self.knowledge_items[knowledge.id] = knowledge
        self.vector_store.add_items([knowledge])
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[KnowledgeItem, float]]:
        """Search for relevant knowledge"""
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        query_embedding = self.embedding_model.encode_single(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def get_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID"""
        return self.knowledge_items.get(knowledge_id)


class ApplicationCorpus:
    """Manages the application corpus with many-to-many links"""
    
    def __init__(self, config: RAGPlusConfig):
        self.config = config
        self.applications: Dict[str, ApplicationExample] = {}
        self.knowledge_to_apps: Dict[str, List[str]] = {}  # Many-to-many: knowledge -> apps
        self.app_to_knowledge: Dict[str, List[str]] = {}   # Many-to-many: app -> knowledge
        
        # Initialize embedding model based on type
        if config.embedding_type == "openai":
            self.embedding_model = OpenAIEmbedding(config.embedding_model)
        else:
            self.embedding_model = SentenceTransformerEmbedding(config.embedding_model)
        
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.get_dimension())
    
    def add_application(self, application: ApplicationExample, knowledge_ids: List[str] = None):
        """Add application example to corpus with many-to-many links"""
        if application.embedding is None:
            application.embedding = self.embedding_model.encode_single(application.content)
        
        self.applications[application.id] = application
        self.vector_store.add_items([application])
        
        # Use provided knowledge_ids or the application's knowledge_id
        linked_knowledge_ids = knowledge_ids or [application.knowledge_id]
        
        # Add many-to-many links
        for knowledge_id in linked_knowledge_ids:
            if knowledge_id not in self.knowledge_to_apps:
                self.knowledge_to_apps[knowledge_id] = []
            if application.id not in self.knowledge_to_apps[knowledge_id]:
                self.knowledge_to_apps[knowledge_id].append(application.id)
        
        # Add reverse links
        if application.id not in self.app_to_knowledge:
            self.app_to_knowledge[application.id] = []
        for knowledge_id in linked_knowledge_ids:
            if knowledge_id not in self.app_to_knowledge[application.id]:
                self.app_to_knowledge[application.id].append(knowledge_id)
    
    def get_applications_for_knowledge(self, knowledge_id: str, top_k: int = None) -> List[ApplicationExample]:
        """Get applications for a specific knowledge item"""
        if top_k is None:
            top_k = self.config.application_top_k
        
        app_ids = self.knowledge_to_apps.get(knowledge_id, [])
        applications = [self.applications[app_id] for app_id in app_ids]
        return applications[:top_k]
    
    def semantic_search_applications(
        self,
        query: str,
        knowledge_context: str = None,
        top_k: int = None
    ) -> List[Tuple[ApplicationExample, float]]:
        """
        Semantic search for applications with optional knowledge context
        This implements the semantic fallback mechanism
        """
        if top_k is None:
            top_k = self.config.application_top_k
        
        # Create search query with knowledge context if provided
        search_query = query
        if knowledge_context:
            search_query = f"{query} [Context: {knowledge_context}]"
        
        # Search in vector store
        query_embedding = self.embedding_model.encode_single(search_query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def get_applications_with_fallback(
        self,
        knowledge_id: str,
        query: str,
        knowledge_content: str = None,
        top_k: int = None
    ) -> List[ApplicationExample]:
        """
        Get applications for knowledge with semantic fallback
        First tries direct links, then semantic search if needed
        """
        if top_k is None:
            top_k = self.config.application_top_k
        
        # First try to get directly linked applications
        direct_apps = self.get_applications_for_knowledge(knowledge_id, top_k)
        
        # If we have enough direct applications, return them
        if len(direct_apps) >= top_k:
            return direct_apps
        
        # Otherwise, use semantic search to find more applications
        needed = top_k - len(direct_apps)
        knowledge_context = knowledge_content if knowledge_content else ""
        
        semantic_results = self.semantic_search_applications(
            query, knowledge_context, needed
        )
        
        # Add semantic results (avoiding duplicates)
        seen_ids = {app.id for app in direct_apps}
        for app, _ in semantic_results:
            if app.id not in seen_ids:
                direct_apps.append(app)
                seen_ids.add(app.id)
        
        return direct_apps[:top_k]
    
    def update_many_to_many_links(self, knowledge_to_apps_mapping: Dict[str, List[str]]):
        """
        Update many-to-many links based on matching results
        """
        # Clear existing links
        self.knowledge_to_apps.clear()
        self.app_to_knowledge.clear()
        
        # Add new links
        for knowledge_id, app_ids in knowledge_to_apps_mapping.items():
            for app_id in app_ids:
                if app_id in self.applications:
                    # Add knowledge -> app link
                    if knowledge_id not in self.knowledge_to_apps:
                        self.knowledge_to_apps[knowledge_id] = []
                    if app_id not in self.knowledge_to_apps[knowledge_id]:
                        self.knowledge_to_apps[knowledge_id].append(app_id)
                    
                    # Add app -> knowledge link
                    if app_id not in self.app_to_knowledge:
                        self.app_to_knowledge[app_id] = []
                    if knowledge_id not in self.app_to_knowledge[app_id]:
                        self.app_to_knowledge[app_id].append(knowledge_id)


class RAGPlus:
    """Main RAG+ system implementation"""
    
    def __init__(self, config: RAGPlusConfig, llm_client=None):
        self.config = config
        
        # Set OpenAI API key if provided
        if config.openai_api_key:
            import os
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
        
        self.knowledge_corpus = KnowledgeCorpus(config)
        self.application_corpus = ApplicationCorpus(config)
        self.llm_client = llm_client
        
        # Initialize components
        self.application_generator = ApplicationGenerator(llm_client, config) if llm_client else None
        self.application_matcher = ApplicationMatcher(config, self.knowledge_corpus.embedding_model)
        self.reranker = Reranker()
        
        # Domain-specific prompt templates
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load domain-specific prompt templates"""
        return {
            "mathematics": """
            Solve this mathematical problem using the provided knowledge and examples:
            
            {knowledge_and_applications}
            
            Question: {question}
            
            Provide step-by-step reasoning and final answer.
            """,
            
            "legal": """
            As a legal expert, analyze this case using the provided legal knowledge and applications:
            
            {knowledge_and_applications}
            
            Case: {question}
            
            Provide legal analysis and conclusion.
            """,
            
            "medical": """
            As a medical professional, diagnose this case using the provided medical knowledge and examples:
            
            {knowledge_and_applications}
            
            Patient Case: {question}
            
            Provide diagnosis and reasoning.
            """
        }
    
    def build_corpora(self, knowledge_source: str, domain: str, existing_applications: List[ApplicationExample] = None):
        """
        Build knowledge and application corpora with full construction pipeline
        Implements both Application Matching and Generation as described in the paper
        """
        logger.info(f"Building corpora for domain: {domain}")
        
        # Extract knowledge items
        knowledge_items = self._extract_knowledge(knowledge_source, domain)
        
        # Add knowledge to corpus
        for knowledge in knowledge_items:
            self.knowledge_corpus.add_knowledge(knowledge)
        
        # Phase 1: Application Matching (if existing applications are provided)
        if existing_applications:
            logger.info("Running Application Matching phase")
            
            # Add existing applications to corpus
            for app in existing_applications:
                self.application_corpus.add_application(app)
            
            # Match applications to knowledge with category alignment
            knowledge_to_apps = self.application_matcher.match_applications_to_knowledge(
                knowledge_items, existing_applications
            )
            
            # Manual refinement step (heuristic-based for implementation)
            knowledge_to_apps = self.application_matcher.manual_refine_matches(
                knowledge_to_apps,
                self.knowledge_corpus.knowledge_items,
                self.application_corpus.applications
            )
            
            # Update many-to-many links
            self.application_corpus.update_many_to_many_links(knowledge_to_apps)
        
        # Phase 2: Application Generation (backfill for missing pairs)
        if self.application_generator:
            logger.info("Running Application Generation phase for missing pairs")
            
            # Get current knowledge-applications mapping
            current_mapping = self.application_corpus.knowledge_to_apps
            
            # Generate applications for knowledge items with insufficient applications
            generated_apps = self.application_generator.generate_applications_for_missing_pairs(
                knowledge_items, current_mapping
            )
            
            # Add generated applications to corpus
            for app in generated_apps:
                self.application_corpus.add_application(app)
        
        logger.info(f"Built corpora with {len(knowledge_items)} knowledge items and {len(self.application_corpus.applications)} applications")
    
    def _extract_knowledge(self, source: str, domain: str) -> List[KnowledgeItem]:
        """Extract knowledge items from source"""
        # Mock implementation - would parse actual sources
        return [
            KnowledgeItem(
                id=f"knowledge_{i}",
                content=f"Mock knowledge item {i} for {domain}",
                knowledge_type=KnowledgeType.CONCEPTUAL,
                domain=domain
            )
            for i in range(10)
        ]
    
    def retrieve(self, query: str, domain: str, use_reranking: bool = True) -> List[KnowledgeApplicationPair]:
        """
        Joint retrieval of knowledge and applications with semantic fallback and reranking
        Implements enhanced retrieval mechanism from the paper
        """
        # Retrieve relevant knowledge
        knowledge_results = self.knowledge_corpus.search(query)
        
        pairs = []
        for knowledge, score in knowledge_results:
            # Get applications for this knowledge with semantic fallback
            applications = self.application_corpus.get_applications_with_fallback(
                knowledge.id, query, knowledge.content
            )
            
            pair = KnowledgeApplicationPair(
                knowledge=knowledge,
                applications=applications,
                relevance_score=score
            )
            pairs.append(pair)
        
        # Apply reranking if enabled
        if use_reranking and pairs:
            pairs = self.reranker.rerank_pairs(query, pairs, self.config.retrieval_top_k)
        
        return pairs
    
    def generate_response(self, query: str, domain: str, use_reranking: bool = True) -> str:
        """Generate response using RAG+ with enhanced retrieval"""
        # Retrieve knowledge and applications
        pairs = self.retrieve(query, domain, use_reranking)
        
        # Format knowledge and applications
        knowledge_and_apps = self._format_knowledge_applications(pairs)
        
        # Get appropriate prompt template
        template = self.prompt_templates.get(domain, self.prompt_templates["mathematics"])
        
        # Generate prompt
        prompt = template.format(
            knowledge_and_applications=knowledge_and_apps,
            question=query
        )
        
        # Generate response (would use actual LLM)
        response = self._generate_llm_response(prompt)
        
        return response
    
    def get_retrieval_agnostic_response(self, query: str, domain: str, retrieval_method: str = "rag_plus") -> str:
        """
        Generate response using different retrieval methods for comparison
        Implements retrieval-agnostic modularity
        """
        if retrieval_method == "rag_plus":
            return self.generate_response(query, domain)
        elif retrieval_method == "knowledge_only":
            # Standard RAG - knowledge only
            knowledge_results = self.knowledge_corpus.search(query)
            knowledge_text = "\n\n".join([k.content for k, _ in knowledge_results])
            
            template = self.prompt_templates.get(domain, self.prompt_templates["mathematics"])
            prompt = template.format(
                knowledge_and_applications=knowledge_text,
                question=query
            )
            return self._generate_llm_response(prompt)
        elif retrieval_method == "applications_only":
            # Applications only ablation
            # Search applications semantically
            app_results = self.application_corpus.semantic_search_applications(query)
            app_text = "\n\n".join([app.content for app, _ in app_results])
            
            template = self.prompt_templates.get(domain, self.prompt_templates["mathematics"])
            prompt = template.format(
                knowledge_and_applications=app_text,
                question=query
            )
            return self._generate_llm_response(prompt)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
    
    def _format_knowledge_applications(self, pairs: List[KnowledgeApplicationPair]) -> str:
        """Format knowledge and applications for prompt"""
        formatted = []
        
        for i, pair in enumerate(pairs, 1):
            formatted.append(f"Knowledge {i}: {pair.knowledge.content}")
            
            for j, app in enumerate(pair.applications, 1):
                formatted.append(f"Application {i}.{j}: {app.content}")
                formatted.append(f"Q: {app.question}")
                formatted.append(f"A: {app.answer}")
        
        return "\n\n".join(formatted)
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        # This would integrate with actual LLM API
        return f"Mock response for prompt: {prompt[:100]}..."


# Example usage
def main():
    """Example usage of RAG+ system"""
    config = RAGPlusConfig()
    rag_plus = RAGPlus(config)
    
    # Build corpora
    rag_plus.build_corpora("mock_source", "mathematics")
    
    # Generate response
    query = "How to solve this integration problem?"
    response = rag_plus.generate_response(query, "mathematics")
    
    print(f"Query: {query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()