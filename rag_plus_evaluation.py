"""
RAG+ Evaluation Framework
Comprehensive evaluation system for RAG+ implementation across different domains
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from rag_plus_core_implementation import RAGPlus, RAGPlusConfig, KnowledgeApplicationPair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    RESPONSE_TIME = "response_time"
    REASONING_QUALITY = "reasoning_quality"


@dataclass
class EvaluationResult:
    """Results of evaluation"""
    metric: EvaluationMetric
    value: float
    domain: str
    model_name: str
    method: str
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationDataset:
    """Evaluation dataset with questions and ground truth"""
    name: str
    domain: str
    questions: List[str]
    ground_truth_answers: List[str]
    metadata: List[Dict[str, Any]] = field(default_factory=list)


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    @abstractmethod
    def evaluate(self, rag_plus: RAGPlus, dataset: EvaluationDataset) -> List[EvaluationResult]:
        """Evaluate RAG+ on dataset"""
        pass


class AccuracyEvaluator(BaseEvaluator):
    """Evaluates accuracy of responses"""
    
    def __init__(self, method_name: str = "RAG+", use_reranking: bool = True):
        self.method_name = method_name
        self.use_reranking = use_reranking
    
    def evaluate(self, rag_plus: RAGPlus, dataset: EvaluationDataset) -> List[EvaluationResult]:
        """Evaluate accuracy"""
        results = []
        predictions = []
        
        logger.info(f"Evaluating accuracy on {dataset.name} using {self.method_name}")
        
        for i, question in enumerate(dataset.questions):
            try:
                # Use different retrieval methods based on method name
                if self.method_name == "RAG":
                    response = rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "knowledge_only")
                elif self.method_name == "Applications Only":
                    response = rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "applications_only")
                elif self.method_name == "RAG+ (No Rerank)":
                    response = rag_plus.generate_response(question, dataset.domain, use_reranking=False)
                else:  # Default RAG+
                    response = rag_plus.generate_response(question, dataset.domain, use_reranking=self.use_reranking)
                
                predictions.append(response)
                
                # Simple string matching for demo (would use more sophisticated matching)
                is_correct = self._compare_answers(response, dataset.ground_truth_answers[i])
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                predictions.append("")
                is_correct = False
        
        # Calculate accuracy
        correct_count = sum(
            self._compare_answers(pred, truth)
            for pred, truth in zip(predictions, dataset.ground_truth_answers)
        )
        accuracy = correct_count / len(dataset.ground_truth_answers)
        
        results.append(EvaluationResult(
            metric=EvaluationMetric.ACCURACY,
            value=accuracy,
            domain=dataset.domain,
            model_name=rag_plus.config.llm_model,
            method=self.method_name
        ))
        
        return results
    
    def _compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth"""
        # Simple implementation - would use more sophisticated matching
        return ground_truth.lower() in predicted.lower()


class ReasoningQualityEvaluator(BaseEvaluator):
    """Evaluates quality of reasoning"""
    
    def __init__(self, llm_client, method_name: str = "RAG+", use_reranking: bool = True):
        self.llm_client = llm_client
        self.method_name = method_name
        self.use_reranking = use_reranking
    
    def evaluate(self, rag_plus: RAGPlus, dataset: EvaluationDataset) -> List[EvaluationResult]:
        """Evaluate reasoning quality"""
        results = []
        quality_scores = []
        
        logger.info(f"Evaluating reasoning quality on {dataset.name} using {self.method_name}")
        
        for i, question in enumerate(dataset.questions):
            # Use different retrieval methods based on method name
            if self.method_name == "RAG":
                response = rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "knowledge_only")
            elif self.method_name == "Applications Only":
                response = rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "applications_only")
            elif self.method_name == "RAG+ (No Rerank)":
                response = rag_plus.generate_response(question, dataset.domain, use_reranking=False)
            else:  # Default RAG+
                response = rag_plus.generate_response(question, dataset.domain, use_reranking=self.use_reranking)
            
            # Use LLM to evaluate reasoning quality
            quality_score = self._evaluate_reasoning_quality(
                question, response, dataset.ground_truth_answers[i]
            )
            quality_scores.append(quality_score)
        
        avg_quality = np.mean(quality_scores)
        
        results.append(EvaluationResult(
            metric=EvaluationMetric.REASONING_QUALITY,
            value=avg_quality,
            domain=dataset.domain,
            model_name=rag_plus.config.llm_model,
            method=self.method_name
        ))
        
        return results
    
    def _evaluate_reasoning_quality(self, question: str, response: str, ground_truth: str) -> float:
        """Evaluate reasoning quality using LLM"""
        prompt = f"""
        Evaluate the reasoning quality of this response on a scale of 1-10:
        
        Question: {question}
        Response: {response}
        Ground Truth: {ground_truth}
        
        Consider:
        1. Logical coherence
        2. Step-by-step reasoning
        3. Correct use of knowledge
        4. Final answer accuracy
        
        Provide only a numerical score (1-10):
        """
        
        # Mock implementation - would call actual LLM
        return 7.5


class PerformanceEvaluator(BaseEvaluator):
    """Evaluates performance metrics like response time"""
    
    def __init__(self, method_name: str = "RAG+", use_reranking: bool = True):
        self.method_name = method_name
        self.use_reranking = use_reranking
    
    def evaluate(self, rag_plus: RAGPlus, dataset: EvaluationDataset) -> List[EvaluationResult]:
        """Evaluate performance"""
        results = []
        response_times = []
        
        logger.info(f"Evaluating performance on {dataset.name} using {self.method_name}")
        
        for question in dataset.questions:
            start_time = time.time()
            try:
                # Use different retrieval methods based on method name
                if self.method_name == "RAG":
                    rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "knowledge_only")
                elif self.method_name == "Applications Only":
                    rag_plus.get_retrieval_agnostic_response(question, dataset.domain, "applications_only")
                elif self.method_name == "RAG+ (No Rerank)":
                    rag_plus.generate_response(question, dataset.domain, use_reranking=False)
                else:  # Default RAG+
                    rag_plus.generate_response(question, dataset.domain, use_reranking=self.use_reranking)
                
                end_time = time.time()
                response_times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Error in performance evaluation: {e}")
                response_times.append(float('inf'))
        
        avg_response_time = np.mean([t for t in response_times if t != float('inf')])
        
        results.append(EvaluationResult(
            metric=EvaluationMetric.RESPONSE_TIME,
            value=avg_response_time,
            domain=dataset.domain,
            model_name=rag_plus.config.llm_model,
            method=self.method_name
        ))
        
        return results


class RAGPlusEvaluator:
    """Main evaluation framework for RAG+"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.evaluators = None  # Will be created per method
    
    def _create_evaluators(self, method_name: str, use_reranking: bool = True):
        """Create evaluators for specific method"""
        return [
            AccuracyEvaluator(method_name, use_reranking),
            ReasoningQualityEvaluator(self.llm_client, method_name, use_reranking) if self.llm_client else None,
            PerformanceEvaluator(method_name, use_reranking)
        ]
    
    def evaluate_system(
        self, 
        rag_plus: RAGPlus, 
        datasets: List[EvaluationDataset]
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate RAG+ system on multiple datasets"""
        all_results = {}
        
        for dataset in datasets:
            logger.info(f"Evaluating on dataset: {dataset.name}")
            dataset_results = []
            
            for evaluator in self.evaluators:
                try:
                    results = evaluator.evaluate(rag_plus, dataset)
                    dataset_results.extend(results)
                except Exception as e:
                    logger.error(f"Error in evaluator {evaluator.__class__.__name__}: {e}")
            
            all_results[dataset.name] = dataset_results
        
        return all_results
    
    def compare_methods(
        self,
        methods: Dict[str, Any],
        datasets: List[EvaluationDataset]
    ) -> pd.DataFrame:
        """Compare different RAG methods"""
        comparison_results = []
        
        for method_name, method_impl in methods.items():
            # Skip baseline (None) for now
            if method_impl is None:
                continue
                
            # Determine if we should use reranking
            use_reranking = "No Rerank" not in method_name
            
            # Create evaluators for this method
            self.evaluators = self._create_evaluators(method_name, use_reranking)
            
            for dataset in datasets:
                results = self.evaluate_system(method_impl, [dataset])
                
                for result in results[dataset.name]:
                    comparison_results.append({
                        'Method': method_name,
                        'Dataset': dataset.name,
                        'Domain': dataset.domain,
                        'Metric': result.metric.value,
                        'Value': result.value
                    })
        
        return pd.DataFrame(comparison_results)
    
    def generate_report(self, results: Dict[str, List[EvaluationResult]]) -> str:
        """Generate evaluation report"""
        report = []
        report.append("# RAG+ Evaluation Report\n")
        
        for dataset_name, dataset_results in results.items():
            report.append(f"## Dataset: {dataset_name}\n")
            
            # Group by metric
            metrics = {}
            for result in dataset_results:
                if result.metric.value not in metrics:
                    metrics[result.metric.value] = []
                metrics[result.metric.value].append(result.value)
            
            for metric_name, values in metrics.items():
                avg_value = np.mean(values)
                report.append(f"- {metric_name}: {avg_value:.4f}")
            
            report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, results_df: pd.DataFrame, output_path: str = None):
        """Plot evaluation results"""
        plt.figure(figsize=(12, 8))
        
        # Create subplot for each metric
        metrics = results_df['Metric'].unique()
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_data = results_df[results_df['Metric'] == metric]
            
            sns.barplot(
                data=metric_data,
                x='Method',
                y='Value',
                hue='Domain',
                ax=axes[i]
            )
            axes[i].set_title(f'{metric} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


class DatasetLoader:
    """Loads evaluation datasets"""
    
    @staticmethod
    def load_mathqa_dataset(file_path: str) -> EvaluationDataset:
        """Load MathQA dataset"""
        # Mock implementation - would load actual dataset
        return EvaluationDataset(
            name="MathQA",
            domain="mathematics",
            questions=[
                "Find the derivative of f(x) = x^2 + 3x - 5",
                "Solve the integral ∫(2x + 1)dx",
                "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1"
            ],
            ground_truth_answers=[
                "f'(x) = 2x + 3",
                "x^2 + x + C",
                "2"
            ]
        )
    
    @staticmethod
    def load_legal_dataset(file_path: str) -> EvaluationDataset:
        """Load legal dataset"""
        return EvaluationDataset(
            name="Legal Sentencing",
            domain="legal",
            questions=[
                "What is the appropriate sentence for intentional injury causing minor harm?",
                "How does self-defense affect sentencing in assault cases?"
            ],
            ground_truth_answers=[
                "Up to 3 years imprisonment",
                "May reduce or eliminate criminal liability"
            ]
        )
    
    @staticmethod
    def load_medical_dataset(file_path: str) -> EvaluationDataset:
        """Load medical dataset"""
        return EvaluationDataset(
            name="MedQA",
            domain="medical",
            questions=[
                "What are the symptoms of myocardial infarction?",
                "How is hypertension diagnosed?"
            ],
            ground_truth_answers=[
                "Chest pain, shortness of breath, sweating",
                "Blood pressure measurement >140/90 mmHg"
            ]
        )


def create_baseline_methods(config: RAGPlusConfig) -> Dict[str, Any]:
    """Create baseline methods for comparison"""
    from rag_plus_core_implementation import RAGPlus
    
    # Create RAG+ system
    rag_plus = RAGPlus(config)
    
    # Return different retrieval methods for comparison
    return {
        "Baseline": None,  # Would be vanilla LLM
        "RAG": rag_plus,  # Standard RAG (knowledge only)
        "RAG+": rag_plus, # Full RAG+ implementation
        "RAG+ (No Rerank)": rag_plus,  # RAG+ without reranking
        "Applications Only": rag_plus  # Applications only ablation
    }


def main():
    """Main evaluation pipeline with comprehensive comparison"""
    logger.info("Starting RAG+ evaluation")
    
    # Initialize evaluator
    evaluator = RAGPlusEvaluator()
    
    # Load datasets
    datasets = [
        DatasetLoader.load_mathqa_dataset("mock_mathqa.json"),
        DatasetLoader.load_legal_dataset("mock_legal.json"),
        DatasetLoader.load_medical_dataset("mock_medical.json")
    ]
    
    # Create configuration
    config = RAGPlusConfig()
    
    # Create baseline methods
    methods = create_baseline_methods(config)
    
    # Build corpora for RAG+ system
    rag_plus = methods["RAG+"]
    for dataset in datasets:
        rag_plus.build_corpora(f"mock_source_{dataset.domain}", dataset.domain)
    
    # Compare methods
    comparison_results = evaluator.compare_methods(methods, datasets)
    
    # Generate report
    print("\n=== RAG+ Evaluation Results ===")
    
    # Group by method and metric
    method_metrics = {}
    for _, row in comparison_results.iterrows():
        method = row['Method']
        metric = row['Metric']
        value = row['Value']
        
        if method not in method_metrics:
            method_metrics[method] = {}
        if metric not in method_metrics[method]:
            method_metrics[method][metric] = []
        
        method_metrics[method][metric].append(value)
    
    # Calculate averages
    for method, metrics in method_metrics.items():
        print(f"\n{method}:")
        for metric, values in metrics.items():
            avg_value = sum(values) / len(values)
            print(f"  {metric}: {avg_value:.4f}")
    
    # Plot results
    evaluator.plot_results(comparison_results, "rag_plus_evaluation_results.png")
    
    # Save results
    comparison_results.to_csv("rag_plus_evaluation_results.csv", index=False)
    
    # Save detailed results
    all_results = {}
    for method_name, method_impl in methods.items():
        if method_impl is not None:
            method_results = evaluator.evaluate_system(method_impl, datasets)
            all_results[method_name] = method_results
    
    with open("detailed_evaluation_results.json", "w") as f:
        # Convert results to serializable format
        serializable_results = {}
        for method_name, method_results in all_results.items():
            serializable_results[method_name] = {}
            for dataset_name, dataset_results in method_results.items():
                serializable_results[method_name][dataset_name] = [
                    {
                        'metric': result.metric.value,
                        'value': result.value,
                        'domain': result.domain,
                        'model_name': result.model_name,
                        'method': result.method
                    }
                    for result in dataset_results
                ]
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Evaluation completed successfully")
    logger.info("Results saved to rag_plus_evaluation_results.csv and detailed_evaluation_results.json")


if __name__ == "__main__":
    main()