"""
RAG+ Integration with Existing Frameworks
Integration layer for connecting RAG+ with popular RAG frameworks like LangChain and LlamaIndex
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import json

from rag_plus_core_implementation import RAGPlus, RAGPlusConfig, KnowledgeApplicationPair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAGFramework(ABC):
    """Abstract base class for RAG framework integrations"""
    
    @abstractmethod
    def create_retriever(self, config: Dict[str, Any]):
        """Create retriever for the framework"""
        pass
    
    @abstractmethod
    def create_chain(self, retriever, prompt_template: str):
        """Create chain for the framework"""
        pass


class LangChainIntegration(BaseRAGFramework):
    """Integration with LangChain framework"""
    
    def __init__(self):
        self.langchain_available = self._check_langchain()
    
    def _check_langchain(self) -> bool:
        """Check if LangChain is available"""
        try:
            import langchain
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain.llms import OpenAI
            return True
        except ImportError:
            logger.warning("LangChain not available. Install with: pip install langchain")
            return False
    
    def create_retriever(self, config: Dict[str, Any]):
        """Create LangChain retriever"""
        if not self.langchain_available:
            return None
        
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import FAISS
            from langchain.docstore import InMemoryDocstore
            from langchain.retrievers import VectorStoreRetriever
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
            
            # Create vector store (would use actual documents)
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=FAISS.create_index(None, 384),  # Mock index
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            
            # Create retriever
            retriever = VectorStoreRetriever(
                vectorstore=vectorstore,
                search_kwargs={"k": config.get("top_k", 3)}
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating LangChain retriever: {e}")
            return None
    
    def create_chain(self, retriever, prompt_template: str):
        """Create LangChain chain"""
        if not self.langchain_available or retriever is None:
            return None
        
        try:
            from langchain.chains import RetrievalQA
            from langchain.llms import OpenAI
            
            # Create LLM
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.0
            )
            
            # Create chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"Error creating LangChain chain: {e}")
            return None
    
    def integrate_rag_plus(self, rag_plus: RAGPlus) -> Any:
        """Integrate RAG+ with LangChain"""
        if not self.langchain_available:
            return None
        
        # Create custom retriever that uses RAG+
        class RAGPlusRetriever:
            def __init__(self, rag_plus_system):
                self.rag_plus = rag_plus_system
            
            def get_relevant_documents(self, query: str):
                # Use RAG+ to retrieve knowledge and applications
                pairs = self.rag_plus.retrieve(query, "mathematics")  # Default domain
                
                # Convert to LangChain documents
                documents = []
                for pair in pairs:
                    # Add knowledge document
                    documents.append({
                        'page_content': pair.knowledge.content,
                        'metadata': {
                            'type': 'knowledge',
                            'id': pair.knowledge.id,
                            'domain': pair.knowledge.domain
                        }
                    })
                    
                    # Add application documents
                    for app in pair.applications:
                        documents.append({
                            'page_content': app.content,
                            'metadata': {
                                'type': 'application',
                                'id': app.id,
                                'knowledge_id': app.knowledge_id
                            }
                        })
                
                return documents
        
        return RAGPlusRetriever(rag_plus)


class LlamaIndexIntegration(BaseRAGFramework):
    """Integration with LlamaIndex framework"""
    
    def __init__(self):
        self.llamaindex_available = self._check_llamaindex()
    
    def _check_llamaindex(self) -> bool:
        """Check if LlamaIndex is available"""
        try:
            import llama_index
            from llama_index import VectorStoreIndex, SimpleDirectoryReader
            from llama_index.llms import OpenAI
            from llama_index.embeddings import HuggingFaceEmbedding
            return True
        except ImportError:
            logger.warning("LlamaIndex not available. Install with: pip install llama-index")
            return False
    
    def create_retriever(self, config: Dict[str, Any]):
        """Create LlamaIndex retriever"""
        if not self.llamaindex_available:
            return None
        
        try:
            from llama_index import VectorStoreIndex, Document
            from llama_index.embeddings import HuggingFaceEmbedding
            
            # Create embeddings
            embed_model = HuggingFaceEmbedding(
                model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
            
            # Create documents (would use actual documents)
            documents = [
                Document(text="Mock document 1"),
                Document(text="Mock document 2")
            ]
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents, 
                embed_model=embed_model
            )
            
            # Create retriever
            retriever = index.as_retriever(
                similarity_top_k=config.get("top_k", 3)
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex retriever: {e}")
            return None
    
    def create_chain(self, retriever, prompt_template: str):
        """Create LlamaIndex query engine"""
        if not self.llamaindex_available or retriever is None:
            return None
        
        try:
            from llama_index.llms import OpenAI
            from llama_index import ServiceContext
            
            # Create LLM
            llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.0
            )
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm
            )
            
            # Create query engine
            query_engine = retriever.as_query_engine(
                service_context=service_context,
                text_qa_template=prompt_template
            )
            
            return query_engine
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex chain: {e}")
            return None
    
    def integrate_rag_plus(self, rag_plus: RAGPlus) -> Any:
        """Integrate RAG+ with LlamaIndex"""
        if not self.llamaindex_available:
            return None
        
        # Create custom query engine that uses RAG+
        class RAGPlusQueryEngine:
            def __init__(self, rag_plus_system):
                self.rag_plus = rag_plus_system
            
            def query(self, query_str: str):
                # Use RAG+ to generate response
                response = self.rag_plus.generate_response(query_str, "mathematics")
                
                # Return LlamaIndex response object
                class MockResponse:
                    def __init__(self, response_text):
                        self.response = response_text
                
                return MockResponse(response)
        
        return RAGPlusQueryEngine(rag_plus)


class RAGPlusFrameworkAdapter:
    """Adapter for integrating RAG+ with multiple frameworks"""
    
    def __init__(self):
        self.frameworks = {
            "langchain": LangChainIntegration(),
            "llamaindex": LlamaIndexIntegration()
        }
    
    def create_integration(self, framework_name: str, rag_plus: RAGPlus) -> Any:
        """Create integration with specified framework"""
        if framework_name not in self.frameworks:
            raise ValueError(f"Unsupported framework: {framework_name}")
        
        framework = self.frameworks[framework_name]
        return framework.integrate_rag_plus(rag_plus)
    
    def create_enhanced_rag(
        self, 
        framework_name: str, 
        rag_plus: RAGPlus,
        config: Dict[str, Any]
    ) -> Any:
        """Create enhanced RAG system with framework integration"""
        if framework_name not in self.frameworks:
            raise ValueError(f"Unsupported framework: {framework_name}")
        
        framework = self.frameworks[framework_name]
        
        # Create retriever
        retriever = framework.create_retriever(config)
        
        # Create prompt template
        prompt_template = self._create_enhanced_prompt_template(config)
        
        # Create chain/query engine
        chain = framework.create_chain(retriever, prompt_template)
        
        return chain
    
    def _create_enhanced_prompt_template(self, config: Dict[str, Any]) -> str:
        """Create enhanced prompt template for RAG+"""
        domain = config.get("domain", "mathematics")
        
        templates = {
            "mathematics": """
            Use the following knowledge and application examples to solve the mathematical problem:
            
            {context}
            
            Question: {question}
            
            Provide step-by-step reasoning and the final answer.
            """,
            
            "legal": """
            As a legal expert, analyze this case using the provided legal knowledge and examples:
            
            {context}
            
            Case: {question}
            
            Provide legal analysis and conclusion.
            """,
            
            "medical": """
            As a medical professional, diagnose this case using the provided medical knowledge and examples:
            
            {context}
            
            Patient Case: {question}
            
            Provide diagnosis and reasoning.
            """
        }
        
        return templates.get(domain, templates["mathematics"])


class RAGPlusAPI:
    """REST API for RAG+ system"""
    
    def __init__(self, rag_plus: RAGPlus):
        self.rag_plus = rag_plus
        self.app = None
        self._setup_flask()
    
    def _setup_flask(self):
        """Setup Flask application"""
        try:
            from flask import Flask, request, jsonify
            
            self.app = Flask(__name__)
            
            @self.app.route('/query', methods=['POST'])
            def query():
                """Handle query requests"""
                try:
                    data = request.get_json()
                    query = data.get('query')
                    domain = data.get('domain', 'mathematics')
                    
                    if not query:
                        return jsonify({'error': 'Query is required'}), 400
                    
                    response = self.rag_plus.generate_response(query, domain)
                    
                    return jsonify({
                        'query': query,
                        'response': response,
                        'domain': domain
                    })
                    
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            @self.app.route('/retrieve', methods=['POST'])
            def retrieve():
                """Handle retrieval requests"""
                try:
                    data = request.get_json()
                    query = data.get('query')
                    domain = data.get('domain', 'mathematics')
                    
                    if not query:
                        return jsonify({'error': 'Query is required'}), 400
                    
                    pairs = self.rag_plus.retrieve(query, domain)
                    
                    # Convert to serializable format
                    results = []
                    for pair in pairs:
                        result = {
                            'knowledge': {
                                'id': pair.knowledge.id,
                                'content': pair.knowledge.content,
                                'type': pair.knowledge.knowledge_type.value,
                                'domain': pair.knowledge.domain
                            },
                            'applications': [
                                {
                                    'id': app.id,
                                    'content': app.content,
                                    'question': app.question,
                                    'answer': app.answer
                                }
                                for app in pair.applications
                            ],
                            'relevance_score': pair.relevance_score
                        }
                        results.append(result)
                    
                    return jsonify({
                        'query': query,
                        'results': results,
                        'domain': domain
                    })
                    
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            @self.app.route('/health', methods=['GET'])
            def health():
                """Health check endpoint"""
                return jsonify({'status': 'healthy'})
            
        except ImportError:
            logger.warning("Flask not available. Install with: pip install flask")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        if self.app is None:
            raise RuntimeError("Flask not available")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Example usage of RAG+ integrations"""
    # Create RAG+ system
    config = RAGPlusConfig()
    rag_plus = RAGPlus(config)
    
    # Build corpora
    rag_plus.build_corpora("mock_source", "mathematics")
    
    # Create framework adapter
    adapter = RAGPlusFrameworkAdapter()
    
    # Test LangChain integration
    try:
        langchain_integration = adapter.create_integration("langchain", rag_plus)
        if langchain_integration:
            logger.info("LangChain integration created successfully")
    except Exception as e:
        logger.error(f"LangChain integration failed: {e}")
    
    # Test LlamaIndex integration
    try:
        llamaindex_integration = adapter.create_integration("llamaindex", rag_plus)
        if llamaindex_integration:
            logger.info("LlamaIndex integration created successfully")
    except Exception as e:
        logger.error(f"LlamaIndex integration failed: {e}")
    
    # Create API
    api = RAGPlusAPI(rag_plus)
    
    # Test query
    query = "How to solve this integration problem?"
    response = rag_plus.generate_response(query, "mathematics")
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Run API (uncomment to start server)
    # api.run(debug=True)


if __name__ == "__main__":
    main()