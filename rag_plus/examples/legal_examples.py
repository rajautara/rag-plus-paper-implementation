"""
Legal Domain Examples for RAG+

This module demonstrates RAG+ usage for legal analysis,
case reasoning, and sentencing prediction.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_plus import (
    RAGPlus,
    KnowledgeItem,
    OpenAILLM,
    OpenAIEmbeddingModel,
    SimpleEmbeddingModel
)


class MockLegalLLM:
    """Mock LLM for legal examples when no API key available."""

    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        if "article 234" in prompt.lower() or "intentional injury" in prompt.lower():
            return """Question: What is the sentencing range for intentional injury causing first-degree minor injuries?
Answer: According to Article 234 of the Criminal Law:
- Basic penalty: Up to 3 years imprisonment, criminal detention, or public surveillance
- Mitigating factors: First offense, voluntary surrender, victim compensation
- For first-degree minor injuries with voluntary surrender: Typically 0-36 months (Answer: A)"""
        elif "theft" in prompt.lower():
            return """Question: What constitutes theft and what are the sentencing guidelines?
Answer: Article 264: Theft of public or private property in relatively large amounts.
Sentencing considers: Amount stolen, prior offenses, use of violence, burglary.
Small amounts: Up to 3 years. Large amounts: 3-10 years. Huge amounts: 10+ years."""
        else:
            return "Question: Legal question\nAnswer: Legal analysis and conclusion"


def example_criminal_sentencing():
    """Example: Criminal sentencing prediction"""
    print("\n" + "="*80)
    print("LEGAL - CRIMINAL SENTENCING")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
        print("Using OpenAI GPT-3.5-turbo")
    else:
        llm = MockLegalLLM()
        embedding_model = SimpleEmbeddingModel()
        print("Using Mock LLM (set OPENAI_API_KEY for real results)")

    rag_plus = RAGPlus(llm, embedding_model, top_k=3)

    knowledge_items = [
        KnowledgeItem(
            id="crim_001",
            content="""Article 234 - Intentional Injury:
Whoever intentionally harms the body of another person shall be sentenced:
- Basic offense: Up to 3 years imprisonment, criminal detention, or public surveillance
- Serious injury: 3-10 years imprisonment
- Death or serious disability with cruel means: 10+ years, life, or death penalty
Severity determined by injury assessment report.""",
            knowledge_type="procedural",
            metadata={"category": "criminal_law", "article": "234"}
        ),
        KnowledgeItem(
            id="crim_002",
            content="""Sentencing Factors for Intentional Injury:
Mitigating: First offense, voluntary surrender, victim compensation, reconciliation, provocation
Aggravating: Weapons, multiple victims, vulnerable victims (children/elderly), premeditation
Minor injury (1st degree) + mitigating factors: Often 0-36 months
Minor injury + aggravating factors: 36-120 months possible""",
            knowledge_type="procedural",
            metadata={"category": "criminal_law", "subcategory": "sentencing"}
        ),
        KnowledgeItem(
            id="crim_003",
            content="""Criminal Procedure - Voluntary Surrender:
Turning oneself in after crime is significant mitigating factor.
Requirements: Voluntary (not compelled), truthful confession, willingness to accept punishment
Effect: Can reduce sentence by 1-2 levels or more for first-time offenders
Often combined with other mitigating factors for lenient sentences.""",
            knowledge_type="procedural",
            metadata={"category": "criminal_procedure"}
        )
    ]

    print("Building legal knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} case application examples\n")

    # Sentencing prediction case from RAG+ paper
    query = """A person intentionally injured another with a weapon, causing first-degree minor injuries.
The defendant has no prior criminal record and voluntarily surrendered to police.
What is the appropriate sentence range?
A) Less than or equal to 36 months
B) Greater than 36 months and less than or equal to 120 months
C) Greater than 120 months"""

    print(f"Legal Case:\n{query}")
    print("-" * 80)
    analysis = rag_plus.generate(query, task_type="legal")
    print(f"Legal Analysis:\n{analysis}\n")


def example_contract_law():
    """Example: Contract law interpretation"""
    print("\n" + "="*80)
    print("LEGAL - CONTRACT LAW")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockLegalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="contract_001",
            content="""Elements of Valid Contract:
1. Offer: Clear proposal by one party
2. Acceptance: Unqualified agreement to offer
3. Consideration: Exchange of value between parties
4. Capacity: Legal ability to enter contract
5. Legality: Contract purpose must be legal
Missing any element makes contract void or voidable.""",
            knowledge_type="conceptual",
            metadata={"category": "contract_law"}
        ),
        KnowledgeItem(
            id="contract_002",
            content="""Breach of Contract Remedies:
Damages: Compensatory (actual losses), consequential (foreseeable), punitive (rare)
Specific Performance: Court orders party to perform (for unique goods/real estate)
Rescission: Cancel contract, return parties to pre-contract position
Restitution: Return benefit conferred to prevent unjust enrichment
Choice depends on breach severity and subject matter.""",
            knowledge_type="procedural",
            metadata={"category": "contract_law", "subcategory": "remedies"}
        )
    ]

    print("Building contract law knowledge...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    queries = [
        "A 16-year-old signs a contract to buy a car. Is the contract enforceable?",
        "Party A breaches contract to sell unique artwork to Party B. What remedy should Party B seek?"
    ]

    for query in queries:
        print(f"Legal Question: {query}")
        print("-" * 80)
        answer = rag_plus.generate(query, task_type="legal")
        print(f"Legal Analysis: {answer}\n")


def example_tort_law():
    """Example: Tort law and liability"""
    print("\n" + "="*80)
    print("LEGAL - TORT LAW")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockLegalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="tort_001",
            content="""Negligence Elements (all required):
1. Duty: Defendant owed plaintiff duty of care
2. Breach: Defendant breached that duty
3. Causation: Breach caused plaintiff's injury (actual + proximate)
4. Damages: Plaintiff suffered actual harm
Standard: Reasonable person under similar circumstances
Common defenses: Contributory/comparative negligence, assumption of risk""",
            knowledge_type="procedural",
            metadata={"category": "tort_law", "subcategory": "negligence"}
        ),
        KnowledgeItem(
            id="tort_002",
            content="""Strict Liability:
Liability without fault for certain activities/products.
Applies to: Abnormally dangerous activities, wild animals, defective products
Elements: (1) Activity/product within scope, (2) Caused harm, (3) Plaintiff harmed
Defenses limited: Plaintiff assumption of risk, misuse of product
No need to prove negligence or intent.""",
            knowledge_type="conceptual",
            metadata={"category": "tort_law", "subcategory": "strict_liability"}
        )
    ]

    print("Building tort law knowledge...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    query = """A driver texting while driving runs a red light and hits a pedestrian in crosswalk.
Pedestrian suffers broken leg requiring surgery. Analyze liability."""

    print(f"Legal Scenario: {query}")
    print("-" * 80)
    answer = rag_plus.generate(query, task_type="legal")
    print(f"Liability Analysis: {answer}\n")


def example_constitutional_law():
    """Example: Constitutional law issues"""
    print("\n" + "="*80)
    print("LEGAL - CONSTITUTIONAL LAW")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockLegalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="const_001",
            content="""Judicial Review Standards:
Strict Scrutiny: Fundamental rights, suspect classifications (race, national origin)
- Government must prove compelling interest + narrowly tailored means
Intermediate Scrutiny: Gender, legitimacy classifications
- Government must prove important interest + substantially related means
Rational Basis: All other classifications
- Law must be rationally related to legitimate government interest""",
            knowledge_type="conceptual",
            metadata={"category": "constitutional_law", "subcategory": "equal_protection"}
        ),
        KnowledgeItem(
            id="const_002",
            content="""Freedom of Speech - First Amendment:
Protected: Political speech, commercial speech, symbolic speech
Unprotected: Incitement, true threats, obscenity, defamation, fighting words
Content-based restrictions: Strict scrutiny
Content-neutral restrictions: Intermediate scrutiny (time/place/manner)
Public forum analysis affects level of protection.""",
            knowledge_type="conceptual",
            metadata={"category": "constitutional_law", "subcategory": "first_amendment"}
        )
    ]

    print("Building constitutional law knowledge...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    query = """A state law prohibits political protests within 1000 feet of any government building.
Is this constitutional?"""

    print(f"Constitutional Question: {query}")
    print("-" * 80)
    answer = rag_plus.generate(query, task_type="legal")
    print(f"Constitutional Analysis: {answer}\n")


def main():
    """Run all legal examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RAG+ LEGAL DOMAIN EXAMPLES                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        example_criminal_sentencing()
        example_contract_law()
        example_tort_law()
        example_constitutional_law()

        print("\n" + "="*80)
        print("All legal examples completed!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
