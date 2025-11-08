"""
RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning

Implementation based on the paper:
"RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning"
by Yu Wang et al.

This module implements the RAG+ framework which extends standard RAG by incorporating
application-aware reasoning through aligned application examples.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a single knowledge item in the corpus."""
    id: str
    content: str
    knowledge_type: str  # 'conceptual' or 'procedural'
    metadata: Optional[Dict] = None


@dataclass
class ApplicationExample:
    """Represents an application example aligned with knowledge."""
    id: str
    knowledge_id: str
    question: str
    answer: str
    reasoning_steps: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Generate text from the LLM."""
        pass


class SimpleEmbeddingModel(EmbeddingModel):
    """Simple embedding model using sentence transformers or similar."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model_name = model_name
        logger.info(f"Initialized SimpleEmbeddingModel with {model_name}")
        # In practice, you would load the actual model here
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        # Placeholder implementation - replace with actual embedding model
        logger.info(f"Generating embeddings for {len(texts)} texts")
        # return self.model.encode(texts)
        # For now, return random embeddings as placeholder
        return np.random.randn(len(texts), 384)


class OpenAILLM(LLMInterface):
    """OpenAI LLM implementation using the OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenAI LLM interface.

        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            model: Model name (default: gpt-3.5-turbo)
            organization: OpenAI organization ID (optional)
            base_url: Base URL for API calls (optional, for custom endpoints)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it with: pip install openai>=1.0.0"
            )

        self.model = model

        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if organization:
            client_kwargs['organization'] = organization
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)
        logger.info(f"Initialized OpenAI LLM with model: {model}")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """
        Generate text from the OpenAI LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            logger.info(f"Generating response with OpenAI model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            generated_text = response.choices[0].message.content
            logger.info(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model using the OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        organization: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenAI embedding model.

        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            model: Embedding model name (default: text-embedding-3-small)
            organization: OpenAI organization ID (optional)
            base_url: Base URL for API calls (optional, for custom endpoints)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it with: pip install openai>=1.0.0"
            )

        self.model = model

        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if organization:
            client_kwargs['organization'] = organization
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)

        # Get embedding dimensions based on model
        self.dimensions = self._get_embedding_dimensions(model)
        logger.info(f"Initialized OpenAI Embedding Model: {model} (dim={self.dimensions})")

    def _get_embedding_dimensions(self, model: str) -> int:
        """Get the embedding dimensions for a given model."""
        # Mapping of known OpenAI embedding models to their dimensions
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions_map.get(model, 1536)  # Default to 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")

            # OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            embeddings_array = np.array(embeddings)

            logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
            return embeddings_array

        except Exception as e:
            logger.error(f"Error generating embeddings from OpenAI: {e}")
            raise


class ApplicationCorpusConstructor:
    """
    Constructs the application corpus aligned with knowledge corpus.
    Implements both application generation and application matching strategies.
    """

    def __init__(self, llm: LLMInterface, embedding_model: EmbeddingModel):
        """
        Initialize the constructor.

        Args:
            llm: LLM interface for generating applications
            embedding_model: Model for generating embeddings
        """
        self.llm = llm
        self.embedding_model = embedding_model
        logger.info("Initialized ApplicationCorpusConstructor")

    def generate_application(self, knowledge: KnowledgeItem) -> ApplicationExample:
        """
        Generate an application example for a given knowledge item.

        Args:
            knowledge: The knowledge item to generate an application for

        Returns:
            ApplicationExample: Generated application example
        """
        if knowledge.knowledge_type == "conceptual":
            prompt = self._create_conceptual_prompt(knowledge)
        elif knowledge.knowledge_type == "procedural":
            prompt = self._create_procedural_prompt(knowledge)
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge.knowledge_type}")

        logger.info(f"Generating application for knowledge ID: {knowledge.id}")
        response = self.llm.generate(prompt, temperature=0.7)

        # Parse the response to extract question and answer
        application = self._parse_application_response(response, knowledge.id)
        return application

    def _create_conceptual_prompt(self, knowledge: KnowledgeItem) -> str:
        """Create prompt for generating conceptual knowledge applications."""
        prompt = f"""Given the following conceptual knowledge, generate a comprehension question or contextual interpretation example that demonstrates understanding of this concept.

Knowledge:
{knowledge.content}

Generate a question-answer pair that helps understand or apply this concept. Format your response as:

Question: [Your question here]
Answer: [Detailed answer with explanation]
"""
        return prompt

    def _create_procedural_prompt(self, knowledge: KnowledgeItem) -> str:
        """Create prompt for generating procedural knowledge applications."""
        prompt = f"""Given the following procedural knowledge, generate a worked example that demonstrates how to apply this procedure step-by-step.

Knowledge:
{knowledge.content}

Generate a problem and its solution that shows how to use this procedure. Format your response as:

Question: [Problem statement]
Answer: [Step-by-step solution with reasoning]
"""
        return prompt

    def _parse_application_response(self, response: str, knowledge_id: str) -> ApplicationExample:
        """Parse LLM response into ApplicationExample."""
        # Simple parsing - in practice, you might use more sophisticated parsing
        lines = response.strip().split('\n')
        question = ""
        answer = ""
        reasoning_steps = []

        current_section = None
        for line in lines:
            if line.startswith("Question:"):
                current_section = "question"
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                current_section = "answer"
                answer = line.replace("Answer:", "").strip()
            elif current_section == "question":
                question += " " + line.strip()
            elif current_section == "answer":
                if line.strip():
                    answer += " " + line.strip()
                    reasoning_steps.append(line.strip())

        app_id = f"app_{knowledge_id}_{hash(question) % 10000}"
        return ApplicationExample(
            id=app_id,
            knowledge_id=knowledge_id,
            question=question,
            answer=answer,
            reasoning_steps=reasoning_steps
        )

    def match_applications(
        self,
        knowledge_items: List[KnowledgeItem],
        real_world_cases: List[Dict],
        temperature: float = 1.0,
        num_votes: int = 3
    ) -> Dict[str, List[str]]:
        """
        Match real-world application cases to knowledge items.
        Uses LLM with temperature sampling and self-consistency voting.

        Args:
            knowledge_items: List of knowledge items
            real_world_cases: List of real-world problem-solution pairs
            temperature: Temperature for LLM sampling
            num_votes: Number of votes for self-consistency

        Returns:
            Dict mapping knowledge_id to list of matched case_ids
        """
        logger.info(f"Matching {len(real_world_cases)} cases to {len(knowledge_items)} knowledge items")

        # Step 1: Categorize knowledge items and cases
        knowledge_categories = self._categorize_items(knowledge_items)
        case_categories = self._categorize_cases(real_world_cases)

        # Step 2: Within each category, match knowledge to cases
        matches = {}
        for category in knowledge_categories:
            if category not in case_categories:
                continue

            cat_knowledge = knowledge_categories[category]
            cat_cases = case_categories[category]

            # Use self-consistency voting for matching
            for knowledge in cat_knowledge:
                matched_cases = self._vote_for_matches(
                    knowledge, cat_cases, temperature, num_votes
                )
                matches[knowledge.id] = matched_cases

        return matches

    def _categorize_items(self, items: List[KnowledgeItem]) -> Dict[str, List[KnowledgeItem]]:
        """Categorize knowledge items using LLM."""
        # Simplified categorization - in practice, use LLM
        categories = {}
        for item in items:
            # Placeholder: extract category from metadata or use LLM
            category = item.metadata.get('category', 'general') if item.metadata else 'general'
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
        return categories

    def _categorize_cases(self, cases: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize real-world cases using LLM."""
        # Simplified categorization
        categories = {}
        for case in cases:
            category = case.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append(case)
        return categories

    def _vote_for_matches(
        self,
        knowledge: KnowledgeItem,
        cases: List[Dict],
        temperature: float,
        num_votes: int
    ) -> List[str]:
        """Use self-consistency voting to match cases to knowledge."""
        # Collect votes from multiple LLM calls with temperature sampling
        all_votes = []

        for _ in range(num_votes):
            prompt = self._create_matching_prompt(knowledge, cases)
            response = self.llm.generate(prompt, temperature=temperature)
            matched_ids = self._parse_matching_response(response)
            all_votes.extend(matched_ids)

        # Count votes and return most voted matches
        vote_counts = {}
        for case_id in all_votes:
            vote_counts[case_id] = vote_counts.get(case_id, 0) + 1

        # Return cases that got at least majority votes
        threshold = num_votes // 2
        matched = [case_id for case_id, count in vote_counts.items() if count >= threshold]
        return matched

    def _create_matching_prompt(self, knowledge: KnowledgeItem, cases: List[Dict]) -> str:
        """Create prompt for matching knowledge to cases."""
        case_descriptions = "\n".join([
            f"{i+1}. ID: {case['id']} - {case.get('description', case.get('question', ''))[:100]}"
            for i, case in enumerate(cases)
        ])

        prompt = f"""Given the following knowledge item, identify which of the listed cases are relevant and would benefit from applying this knowledge.

Knowledge:
{knowledge.content}

Cases:
{case_descriptions}

List the IDs of relevant cases (comma-separated):
"""
        return prompt

    def _parse_matching_response(self, response: str) -> List[str]:
        """Parse matching response to extract case IDs."""
        # Simple parsing - extract IDs from response
        ids = []
        for word in response.replace(',', ' ').split():
            word = word.strip()
            if word.startswith('case_') or word.isdigit():
                ids.append(word)
        return ids


class RAGPlusRetriever:
    """
    Implements the retrieval mechanism for RAG+.
    Retrieves both knowledge and aligned application examples.
    """

    def __init__(self, embedding_model: EmbeddingModel, top_k: int = 3):
        """
        Initialize the retriever.

        Args:
            embedding_model: Model for generating embeddings
            top_k: Number of top results to retrieve
        """
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.knowledge_corpus = []
        self.application_corpus = []
        self.knowledge_embeddings = None
        self.knowledge_to_applications = {}
        logger.info(f"Initialized RAGPlusRetriever with top_k={top_k}")

    def index_knowledge(self, knowledge_items: List[KnowledgeItem]):
        """Index knowledge items for retrieval."""
        logger.info(f"Indexing {len(knowledge_items)} knowledge items")
        self.knowledge_corpus = knowledge_items
        texts = [k.content for k in knowledge_items]
        self.knowledge_embeddings = self.embedding_model.embed(texts)

    def index_applications(self, applications: List[ApplicationExample]):
        """Index application examples."""
        logger.info(f"Indexing {len(applications)} application examples")
        self.application_corpus = applications

        # Build mapping from knowledge_id to applications
        self.knowledge_to_applications = {}
        for app in applications:
            if app.knowledge_id not in self.knowledge_to_applications:
                self.knowledge_to_applications[app.knowledge_id] = []
            self.knowledge_to_applications[app.knowledge_id].append(app)

    def retrieve(self, query: str) -> List[Tuple[KnowledgeItem, ApplicationExample]]:
        """
        Retrieve relevant knowledge-application pairs for a query.

        Args:
            query: The input query

        Returns:
            List of (KnowledgeItem, ApplicationExample) tuples
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        # Embed the query
        query_embedding = self.embedding_model.embed([query])[0]

        # Compute similarities
        similarities = np.dot(self.knowledge_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        # Retrieve knowledge-application pairs
        results = []
        for idx in top_indices:
            knowledge = self.knowledge_corpus[idx]
            # Get aligned application
            applications = self.knowledge_to_applications.get(knowledge.id, [])
            if applications:
                # Take the first application (or implement more sophisticated selection)
                application = applications[0]
                results.append((knowledge, application))
            else:
                logger.warning(f"No application found for knowledge ID: {knowledge.id}")

        return results


class RAGPlus:
    """
    Main RAG+ system implementing application-aware reasoning.
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedding_model: EmbeddingModel,
        top_k: int = 3
    ):
        """
        Initialize RAG+ system.

        Args:
            llm: LLM interface for generation
            embedding_model: Embedding model for retrieval
            top_k: Number of top results to retrieve
        """
        self.llm = llm
        self.constructor = ApplicationCorpusConstructor(llm, embedding_model)
        self.retriever = RAGPlusRetriever(embedding_model, top_k)
        logger.info("Initialized RAG+ system")

    def build_corpus(
        self,
        knowledge_items: List[KnowledgeItem],
        real_world_cases: Optional[List[Dict]] = None,
        use_generation: bool = True,
        use_matching: bool = False
    ) -> List[ApplicationExample]:
        """
        Build the application corpus aligned with knowledge corpus.

        Args:
            knowledge_items: List of knowledge items
            real_world_cases: Optional real-world cases for matching
            use_generation: Whether to generate applications
            use_matching: Whether to match real-world cases

        Returns:
            List of ApplicationExample objects
        """
        logger.info("Building application corpus...")
        applications = []

        if use_generation:
            logger.info("Generating applications for knowledge items...")
            for knowledge in knowledge_items:
                try:
                    app = self.constructor.generate_application(knowledge)
                    applications.append(app)
                except Exception as e:
                    logger.error(f"Error generating application for {knowledge.id}: {e}")

        if use_matching and real_world_cases:
            logger.info("Matching real-world cases to knowledge...")
            matches = self.constructor.match_applications(knowledge_items, real_world_cases)

            # Create ApplicationExample objects from matches
            for knowledge_id, case_ids in matches.items():
                for case_id in case_ids:
                    case = next((c for c in real_world_cases if c['id'] == case_id), None)
                    if case:
                        app = ApplicationExample(
                            id=f"matched_{knowledge_id}_{case_id}",
                            knowledge_id=knowledge_id,
                            question=case.get('question', case.get('problem', '')),
                            answer=case.get('answer', case.get('solution', '')),
                            reasoning_steps=case.get('reasoning_steps'),
                            metadata={'matched': True, 'case_id': case_id}
                        )
                        applications.append(app)

        # Index the knowledge and applications
        self.retriever.index_knowledge(knowledge_items)
        self.retriever.index_applications(applications)

        logger.info(f"Built corpus with {len(applications)} applications")
        return applications

    def generate(self, query: str, task_type: str = "general") -> str:
        """
        Generate answer for a query using RAG+ approach.

        Args:
            query: The input query
            task_type: Type of task (math, legal, medical, general)

        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for query: {query[:100]}...")

        # Retrieve knowledge-application pairs
        retrieved_pairs = self.retriever.retrieve(query)

        if not retrieved_pairs:
            logger.warning("No relevant knowledge-application pairs found, using baseline generation")
            prompt = self._create_baseline_prompt(query, task_type)
        else:
            # Create RAG+ prompt with knowledge and applications
            prompt = self._create_ragplus_prompt(query, retrieved_pairs, task_type)

        # Generate answer
        answer = self.llm.generate(prompt, temperature=0.0)
        return answer

    def _create_baseline_prompt(self, query: str, task_type: str) -> str:
        """Create baseline prompt without retrieval."""
        prompt = f"""Please answer the following {task_type} question:

Question: {query}

Answer:"""
        return prompt

    def _create_ragplus_prompt(
        self,
        query: str,
        retrieved_pairs: List[Tuple[KnowledgeItem, ApplicationExample]],
        task_type: str
    ) -> str:
        """
        Create RAG+ prompt with knowledge and application examples.

        Args:
            query: The input query
            retrieved_pairs: Retrieved knowledge-application pairs
            task_type: Type of task

        Returns:
            Formatted prompt string
        """
        # Build reference section with knowledge and applications
        references = []
        for i, (knowledge, application) in enumerate(retrieved_pairs, 1):
            ref = f"""Knowledge Point {i}:
{knowledge.content}

Application of Knowledge Point {i}:
Question: {application.question}
Answer: {application.answer}
"""
            references.append(ref)

        reference_section = "\n\n".join(references)

        # Create task-specific prompt
        if task_type == "math":
            task_instruction = "solve the following mathematical problem step by step"
        elif task_type == "legal":
            task_instruction = "answer the following legal question based on relevant laws and precedents"
        elif task_type == "medical":
            task_instruction = "answer the following medical question with clinical reasoning"
        else:
            task_instruction = "answer the following question"

        prompt = f"""You are provided with relevant knowledge and application examples to help you {task_instruction}.

Reference Knowledge and Applications:
{reference_section}

Now use the reference knowledge and applications provided for guidance (but do not be strictly bound by them), please {task_instruction}:

Question: {query}

Answer:"""

        return prompt

    def save_corpus(self, knowledge_path: str, applications_path: str):
        """Save knowledge and application corpus to files."""
        # Save knowledge corpus
        knowledge_data = [
            {
                'id': k.id,
                'content': k.content,
                'knowledge_type': k.knowledge_type,
                'metadata': k.metadata
            }
            for k in self.retriever.knowledge_corpus
        ]

        with open(knowledge_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)

        # Save application corpus
        app_data = [
            {
                'id': a.id,
                'knowledge_id': a.knowledge_id,
                'question': a.question,
                'answer': a.answer,
                'reasoning_steps': a.reasoning_steps,
                'metadata': a.metadata
            }
            for a in self.retriever.application_corpus
        ]

        with open(applications_path, 'w', encoding='utf-8') as f:
            json.dump(app_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved corpus to {knowledge_path} and {applications_path}")

    def load_corpus(self, knowledge_path: str, applications_path: str):
        """Load knowledge and application corpus from files."""
        # Load knowledge corpus
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)

        knowledge_items = [
            KnowledgeItem(
                id=k['id'],
                content=k['content'],
                knowledge_type=k['knowledge_type'],
                metadata=k.get('metadata')
            )
            for k in knowledge_data
        ]

        # Load application corpus
        with open(applications_path, 'r', encoding='utf-8') as f:
            app_data = json.load(f)

        applications = [
            ApplicationExample(
                id=a['id'],
                knowledge_id=a['knowledge_id'],
                question=a['question'],
                answer=a['answer'],
                reasoning_steps=a.get('reasoning_steps'),
                metadata=a.get('metadata')
            )
            for a in app_data
        ]

        # Index the loaded data
        self.retriever.index_knowledge(knowledge_items)
        self.retriever.index_applications(applications)

        logger.info(f"Loaded corpus from {knowledge_path} and {applications_path}")


def compare_rag_vs_ragplus(
    rag_system: RAGPlus,
    baseline_llm: LLMInterface,
    test_queries: List[Dict],
    task_type: str = "general"
) -> Dict:
    """
    Compare RAG+ performance against baseline.

    Args:
        rag_system: RAG+ system
        baseline_llm: Baseline LLM for comparison
        test_queries: List of test queries with expected answers
        task_type: Type of task

    Returns:
        Dictionary with comparison results
    """
    results = {
        'baseline': [],
        'ragplus': [],
        'queries': []
    }

    for query_dict in test_queries:
        query = query_dict['query']

        # Baseline generation
        baseline_prompt = f"Please answer: {query}"
        baseline_answer = baseline_llm.generate(baseline_prompt)

        # RAG+ generation
        ragplus_answer = rag_system.generate(query, task_type)

        results['queries'].append(query)
        results['baseline'].append(baseline_answer)
        results['ragplus'].append(ragplus_answer)

    return results
