"""
Medical Domain Examples for RAG+

This module demonstrates RAG+ usage for clinical diagnosis,
medical reasoning, and treatment planning.
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


class MockMedicalLLM:
    """Mock LLM for medical examples when no API key available."""

    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        if "leriche" in prompt.lower() or "aortoiliac" in prompt.lower():
            return """Question: What is Leriche syndrome?
Answer: Leriche syndrome is aortoiliac atherosclerosis causing the classic triad:
1) Claudication in buttocks/thighs
2) Absent or diminished femoral pulses
3) Erectile dysfunction in males
Associated with cardiovascular risk factors."""
        elif "anatomy" in prompt.lower():
            return """Question: What is the difference between gross anatomy and microscopic anatomy?
Answer: Gross anatomy studies structures visible to the naked eye (organs, bones, muscles).
Microscopic anatomy (histology) studies cells and tissues using microscopy.
Both are complementary approaches to understanding body structure."""
        else:
            return "Question: Medical question\nAnswer: Clinical reasoning and answer"


def example_vascular_disease():
    """Example: Vascular disease diagnosis"""
    print("\n" + "="*80)
    print("MEDICAL - VASCULAR DISEASE DIAGNOSIS")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
        print("Using OpenAI GPT-3.5-turbo")
    else:
        llm = MockMedicalLLM()
        embedding_model = SimpleEmbeddingModel()
        print("Using Mock LLM (set OPENAI_API_KEY for real results)")

    rag_plus = RAGPlus(llm, embedding_model, top_k=3)

    knowledge_items = [
        KnowledgeItem(
            id="vasc_001",
            content="""Peripheral Artery Disease (PAD): Atherosclerosis in lower extremity arteries.
Symptoms: Claudication (pain with exertion, relieved by rest), weak/absent pulses, skin changes, slow wound healing.
Risk factors: Diabetes, hypertension, smoking, hyperlipidemia, age >50.""",
            knowledge_type="conceptual",
            metadata={"category": "cardiology", "subcategory": "vascular"}
        ),
        KnowledgeItem(
            id="vasc_002",
            content="""Leriche Syndrome (Aortoiliac Atherosclerosis): Occlusion/stenosis of terminal aorta and iliac arteries.
Classic triad:
1. Claudication in buttocks and thighs
2. Absent or diminished femoral pulses
3. Erectile dysfunction in males
Associated with major cardiovascular risk factors.""",
            knowledge_type="conceptual",
            metadata={"category": "cardiology", "subcategory": "vascular"}
        ),
        KnowledgeItem(
            id="vasc_003",
            content="""Acute Limb Ischemia: 6 P's - Pain, Pallor, Pulselessness, Paresthesia, Paralysis, Poikilothermia (coldness).
Requires urgent intervention within 6 hours to prevent tissue loss.
Causes: Embolism (most common), thrombosis, trauma.""",
            knowledge_type="procedural",
            metadata={"category": "vascular_emergency"}
        )
    ]

    print("Building medical knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} clinical application examples\n")

    # Classic case from the RAG+ paper
    query = """A 67-year-old man presents with erectile dysfunction and deep burning buttock pain
when walking that is relieved by rest. Physical exam shows weak femoral pulses bilaterally.
Past medical history: diabetes, hypertension, 40 pack-year smoking history.
What is the most specific etiology?"""

    print(f"Clinical Case:\n{query}")
    print("-" * 80)
    diagnosis = rag_plus.generate(query, task_type="medical")
    print(f"Diagnosis:\n{diagnosis}\n")


def example_anatomy():
    """Example: Anatomy concepts"""
    print("\n" + "="*80)
    print("MEDICAL - ANATOMY")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMedicalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=2)

    knowledge_items = [
        KnowledgeItem(
            id="anat_001",
            content="""Anatomy encompasses both gross anatomy (macroscopic structures visible to naked eye)
and microscopic anatomy (histology - cells and tissues requiring magnification).
Gross: organs, bones, muscles, vessels
Microscopic: cellular structures, tissue organization""",
            knowledge_type="conceptual",
            metadata={"category": "anatomy", "subcategory": "basic_sciences"}
        ),
        KnowledgeItem(
            id="anat_002",
            content="""Anatomical Position: Body erect, face forward, arms at sides with palms forward, feet together.
Directional terms: Superior/Inferior, Anterior/Posterior, Medial/Lateral, Proximal/Distal.
Reference planes: Sagittal, Coronal (frontal), Transverse (horizontal).""",
            knowledge_type="conceptual",
            metadata={"category": "anatomy"}
        )
    ]

    print("Building anatomy knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    query = "Explain the relationship between gross anatomy and microscopic anatomy in medical education."

    print(f"Query: {query}")
    print("-" * 80)
    answer = rag_plus.generate(query, task_type="medical")
    print(f"Answer: {answer}\n")


def example_cardiology():
    """Example: Cardiology diagnosis"""
    print("\n" + "="*80)
    print("MEDICAL - CARDIOLOGY")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMedicalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=3)

    knowledge_items = [
        KnowledgeItem(
            id="cardio_001",
            content="""Acute Myocardial Infarction (MI): Cardiac muscle necrosis due to prolonged ischemia.
STEMI: ST elevation, transmural infarction, requires immediate reperfusion
NSTEMI: No ST elevation, subendocardial infarction
Diagnosis: Troponin elevation + chest pain or ECG changes
Management: Aspirin, anticoagulation, reperfusion (PCI or fibrinolysis for STEMI)""",
            knowledge_type="procedural",
            metadata={"category": "cardiology", "subcategory": "acs"}
        ),
        KnowledgeItem(
            id="cardio_002",
            content="""Heart Failure Classification (NYHA):
Class I: No limitation, ordinary activity doesn't cause symptoms
Class II: Slight limitation, comfortable at rest
Class III: Marked limitation, comfortable only at rest
Class IV: Symptoms at rest, unable to carry out any activity
Guides treatment intensity and prognosis.""",
            knowledge_type="conceptual",
            metadata={"category": "cardiology", "subcategory": "heart_failure"}
        ),
        KnowledgeItem(
            id="cardio_003",
            content="""Atrial Fibrillation Management:
Rate control: Beta-blockers, calcium channel blockers, digoxin
Rhythm control: Cardioversion, antiarrhythmics (amiodarone, flecainide)
Anticoagulation: CHA2DS2-VASc score guides need for anticoagulation
Stroke prevention is paramount in management.""",
            knowledge_type="procedural",
            metadata={"category": "cardiology", "subcategory": "arrhythmia"}
        )
    ]

    print("Building cardiology knowledge corpus...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    queries = [
        "Patient with chest pain, ST elevation in V1-V4, elevated troponin. What is the diagnosis and immediate management?",
        "How do you decide when to anticoagulate a patient with atrial fibrillation?"
    ]

    for query in queries:
        print(f"Clinical Question: {query}")
        print("-" * 80)
        answer = rag_plus.generate(query, task_type="medical")
        print(f"Answer: {answer}\n")


def example_differential_diagnosis():
    """Example: Complex differential diagnosis"""
    print("\n" + "="*80)
    print("MEDICAL - DIFFERENTIAL DIAGNOSIS")
    print("="*80 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAILLM(model="gpt-3.5-turbo")
        embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
    else:
        llm = MockMedicalLLM()
        embedding_model = SimpleEmbeddingModel()

    rag_plus = RAGPlus(llm, embedding_model, top_k=3)

    knowledge_items = [
        KnowledgeItem(
            id="dx_001",
            content="""Acute Abdominal Pain Differential:
RUQ: Cholecystitis, hepatitis, pneumonia
RLQ: Appendicitis, ovarian pathology, ectopic pregnancy
LUQ: Splenic infarct, pancreatitis
LLQ: Diverticulitis, ovarian pathology
Epigastric: MI, pancreatitis, PUD, GERD
Diffuse: Peritonitis, bowel obstruction, mesenteric ischemia""",
            knowledge_type="conceptual",
            metadata={"category": "gastroenterology"}
        ),
        KnowledgeItem(
            id="dx_002",
            content="""Chest Pain Differential (Life-threatening first):
Cardiac: ACS, aortic dissection, pericarditis
Pulmonary: PE, pneumothorax, pneumonia
GI: Esophageal rupture, PUD
Musculoskeletal: Costochondritis, rib fracture
Red flags: Sudden onset, radiation, associated dyspnea, diaphoresis""",
            knowledge_type="conceptual",
            metadata={"category": "emergency_medicine"}
        )
    ]

    print("Building differential diagnosis knowledge...")
    applications = rag_plus.build_corpus(knowledge_items, use_generation=True)
    print(f"Generated {len(applications)} application examples\n")

    query = """65-year-old with sudden severe epigastric pain radiating to back,
nausea, vomiting. Vital signs show BP 92/60, HR 110. Alcohol history present.
What are the top differential diagnoses?"""

    print(f"Clinical Case: {query}")
    print("-" * 80)
    answer = rag_plus.generate(query, task_type="medical")
    print(f"Differential Diagnosis: {answer}\n")


def main():
    """Run all medical examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       RAG+ MEDICAL DOMAIN EXAMPLES                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        example_vascular_disease()
        example_anatomy()
        example_cardiology()
        example_differential_diagnosis()

        print("\n" + "="*80)
        print("All medical examples completed!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
