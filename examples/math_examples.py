"""
Mathematics Domain Examples for RAG+

This module demonstrates RAG+ usage for mathematical problem-solving,
including calculus, algebra, combinatorics, and probability.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_plus import (
    RAGPlus,
    KnowledgeItem,
    OpenAILLM,
    OpenAIEmbeddingModel,
    SimpleEmbeddingModel
)


class MockMathLLM:
    """Mock LLM for math examples when no API key available."""

    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        if "combination" in prompt.lower():
            return """Question: How many ways can we select 3 items from a set of 5 distinct items?
Answer: Using the combination formula C(n,k) = n!/(k!(n-k)!):
C(5,3) = 5!/(3!×2!) = (5×4×3!)/(3!×2×1) = 20/2 = 10 ways"""
        elif "derivative" in prompt.lower() or "power rule" in prompt.lower():
            return """Question: Find the derivative of f(x) = 3x^4 + 2x^2 - 5x + 7
Answer: Using the power rule d/dx(x^n) = n*x^(n-1):
f'(x) = 3×4×x^3 + 2×2×x - 5 = 12x^3 + 4x - 5"""
        elif "chain rule" in prompt.lower():
            return """Question: Find the derivative of f(x) = sin(3x^2)
Answer: Using the chain rule d/dx(f(g(x))) = f'(g(x))×g'(x):
f'(x) = cos(3x^2) × 6x = 6x×cos(3x^2)"""
        else:
            return "Question: Sample math question\nAnswer: Sample solution"


def example_calculus():
    """Example: Calculus problems with RAG+"""
    print("\n" + "="*80)
    print("MATHEMATICS - CALCULUS")
    print("="*80 + "\n")

    # Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
        print("Using OpenAI GPT-3.5-turbo")
    else:
        llm = MockMathLLM()
        embedding_model = SimpleEmbeddingModel()
        print("Using Mock LLM (set OPENAI_API_KEY for real results)")

    rag_plus = RAGPlus(llm, embedding_model, top_k=3)

    # Define calculus knowledge
    knowledge_items = [
        KnowledgeItem(
            id="calc_001",
            content="Power Rule: The derivative of x^n is n*x^(n-1). This applies to any real number n.",
            knowledge_type="procedural",
            metadata={"category": "calculus", "subcategory": "differentiation"}
        ),
        KnowledgeItem(
            id="calc_002",
            content="Chain Rule: For composite functions, d/dx[f(g(x))] = f'(g(x)) * g'(x). The derivative of the outer function times the derivative of the inner function.",
            knowledge_type="procedural",
            metadata={"category": "calculus", "subcategory": "differentiation"}
        ),
        KnowledgeItem(
            id="calc_003",
            content="Product Rule: For the product of two functions, d/dx[f(x)*g(x)] = f'(x)*g(x) + f(x)*g'(x).",
            knowledge_type="procedural",
            metadata={"category": "calculus", "subcategory": "differentiation"}
        ),
        KnowledgeItem(
            id="calc_004",
            content="Integration by Parts: ∫u dv = uv - ∫v du. Useful when integrating products of functions.",
            knowledge_type="procedural",
            metadata={"category": "calculus", "subcategory": "integration"}
        )
    ]

    # Build corpus
    print("Building knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    # Test queries
    queries = [
        "Find the derivative of x^5 + 3x^2 - 7",
        "What is the derivative of sin(2x^3)?",
        "Find the derivative of x^2 * e^x"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)
        answer = rag_plus.generate(query, task_type="math")
        print(f"Answer: {answer}\n")


def example_combinatorics():
    """Example: Combinatorics problems with RAG+"""
    print("\n" + "="*80)
    print("MATHEMATICS - COMBINATORICS")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMathLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="comb_001",
            content="Combination Formula: C(n,k) = n!/(k!(n-k)!). Used when selecting k items from n items without regard to order.",
            knowledge_type="procedural",
            metadata={"category": "combinatorics"}
        ),
        KnowledgeItem(
            id="comb_002",
            content="Permutation Formula: P(n,k) = n!/(n-k)!. Used when selecting k items from n items where order matters.",
            knowledge_type="procedural",
            metadata={"category": "combinatorics"}
        ),
        KnowledgeItem(
            id="comb_003",
            content="Complementary Counting: When constraints make direct counting difficult, calculate total possibilities minus invalid cases. Especially useful for 'at least one' constraints.",
            knowledge_type="procedural",
            metadata={"category": "combinatorics"}
        )
    ]

    print("Building knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    # Classic combinatorics problem from the paper
    query = "How many ways to choose 3 students from 6 boys and 4 girls, with at least one girl?"

    print(f"Query: {query}")
    print("-" * 80)
    answer = rag_plus.generate(query, task_type="math")
    print(f"Answer: {answer}\n")


def example_probability():
    """Example: Probability problems with RAG+"""
    print("\n" + "="*80)
    print("MATHEMATICS - PROBABILITY")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMathLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="prob_001",
            content="Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B). Updates prior probability given new evidence.",
            knowledge_type="procedural",
            metadata={"category": "probability"}
        ),
        KnowledgeItem(
            id="prob_002",
            content="Law of Total Probability: P(A) = Σ P(A|Bi) * P(Bi) for partition {Bi}.",
            knowledge_type="procedural",
            metadata={"category": "probability"}
        ),
        KnowledgeItem(
            id="prob_003",
            content="Independent Events: Events A and B are independent if P(A∩B) = P(A)*P(B).",
            knowledge_type="conceptual",
            metadata={"category": "probability"}
        )
    ]

    print("Building knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    queries = [
        "A medical test is 99% accurate. If 1% of population has the disease, what's the probability someone who tests positive actually has the disease?",
        "If two dice are rolled, what's the probability that the sum is greater than 8?"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)
        answer = rag_plus.generate(query, task_type="math")
        print(f"Answer: {answer}\n")


def example_algebra():
    """Example: Algebra problems with RAG+"""
    print("\n" + "="*80)
    print("MATHEMATICS - ALGEBRA")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMathLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="alg_001",
            content="Quadratic Formula: For ax² + bx + c = 0, x = [-b ± √(b²-4ac)] / 2a",
            knowledge_type="procedural",
            metadata={"category": "algebra"}
        ),
        KnowledgeItem(
            id="alg_002",
            content="Difference of Squares: a² - b² = (a+b)(a-b). Useful for factoring.",
            knowledge_type="procedural",
            metadata={"category": "algebra"}
        ),
        KnowledgeItem(
            id="alg_003",
            content="Completing the Square: Transform ax² + bx + c into a(x-h)² + k form by adding and subtracting (b/2a)².",
            knowledge_type="procedural",
            metadata={"category": "algebra"}
        )
    ]

    print("Building knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    queries = [
        "Solve: 2x² + 5x - 3 = 0",
        "Factor: x² - 16"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)
        answer = rag_plus.generate(query, task_type="math")
        print(f"Answer: {answer}\n")


def main():
    """Run all mathematics examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      RAG+ MATHEMATICS DOMAIN EXAMPLES                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        example_calculus()
        example_combinatorics()
        example_probability()
        example_algebra()

        print("\n" + "="*80)
        print("All mathematics examples completed!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
