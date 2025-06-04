import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict

class QueryExpansionMethod(ABC):
    """Abstract base class for query expansion methods"""
    
    @abstractmethod
    def expand_query(self, query: str) -> str:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class NoExpansion(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        return query
    
    def get_name(self) -> str:
        return "None"

class ExpansionMethod1(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        return query[:-4] + '_1.mid'
    
    def get_name(self) -> str:
        return "Method1"

class ExpansionMethod2(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        # Implement your second expansion method here
        return query[:-4] + '_2.mid'
    
    def get_name(self) -> str:
        return "Method2"

class ExpansionMethod3(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        # Implement your third expansion method here
        return query[:-4] + '_3.mid'
    
    def get_name(self) -> str:
        return "Method3"

class ExpansionMethod4(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        # Implement your fourth expansion method here
        return query[:-4] + '_4.mid'
    
    def get_name(self) -> str:
        return "Method4"

class Retriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        """Retrieve documents for a query
        Returns: List of (doc_id, score) tuples sorted by score desc
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class SparseRetriever(Retriever):
    def __init__(self):
        # Initialize your sparse retriever (e.g., BM25, TF-IDF)
        pass
    
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        # Implement sparse retrieval logic
        # Return list of (doc_id, score) tuples
        results = []  # placeholder
        return results
    
    def get_name(self) -> str:
        return "Sparse"

class DenseRetriever(Retriever):
    def __init__(self):
        # Initialize your dense retriever (e.g., BERT, DPR)
        pass
    
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        # Implement dense retrieval logic
        # Return list of (doc_id, score) tuples
        results = []  # placeholder
        return results
    
    def get_name(self) -> str:
        return "Dense"

from collections import defaultdict
import heapq

class TelescopeRetriever(Retriever):
    def __init__(self, sparse_retriever: SparseRetriever, dense_retriever: DenseRetriever):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
    
    def retrieve(self, query: str, k: int = 1000) -> list[tuple[str, float]]:
        # Step 1: Retrieve candidates using sparse retriever
        sparse_results = self.sparse_retriever.retrieve(query, k)
        
        # Step 2: Get document IDs from sparse retriever
        candidate_docs = [doc_id for doc_id, _ in sparse_results]
        
        # Step 3: Dense retriever re-ranks only the retrieved documents
        dense_results = self.dense_retriever.rerank(query, candidate_docs)
        
        # Sort by dense re-ranking score
        return sorted(dense_results, key=lambda x: x[1], reverse=True)[:k]
    def get_name(self) -> str:
        return "Telescope"

class MAPEvaluator:
    def __init__(self, relevance_judgments: Dict[str, Dict[str, int]]):
        """
        relevance_judgments: {query_id: {doc_id: relevance_score}}
        relevance_score: 0 (not relevant) or 1 (relevant)
        """
        self.relevance_judgments = relevance_judgments
    
    def calculate_ap(self, query_id: str, retrieved_docs: List[str]) -> float:
        """Calculate Average Precision for a single query"""
        if query_id not in self.relevance_judgments:
            return 0.0
        
        relevant_docs = set(doc_id for doc_id, rel in self.relevance_judgments[query_id].items() if rel > 0)
        
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def calculate_map(self, results: Dict[str, List[str]]) -> float:
        """Calculate Mean Average Precision across all queries"""
        ap_scores = []
        
        for query_id, retrieved_docs in results.items():
            ap = self.calculate_ap(query_id, retrieved_docs)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0

class IRExperiment:
    def __init__(self, queries: Dict[str, str], relevance_judgments: Dict[str, Dict[str, int]]):
        """
        queries: {query_id: query_text}
        relevance_judgments: {query_id: {doc_id: relevance_score}}
        """
        self.queries = queries
        self.evaluator = MAPEvaluator(relevance_judgments)
        
        # Initialize expansion methods
        self.expansion_methods = [
            NoExpansion(),
            ExpansionMethod1(),
            ExpansionMethod2(),
            ExpansionMethod3(),
            ExpansionMethod4()
        ]
        
        # Initialize retrievers
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever()
        self.retrievers = [
            self.sparse_retriever,
            self.dense_retriever,
            TelescopeRetriever(self.sparse_retriever, self.dense_retriever)
        ]
    
    def run_single_experiment(self, expansion_method: QueryExpansionMethod, 
                            retriever: Retriever, k: int = 1000) -> Tuple[float, Dict[str, List[str]]]:
        """Run experiment with specific expansion method and retriever"""
        results = {}
        
        for query_id, original_query in self.queries.items():
            # Expand query
            expanded_query = expansion_method.expand_query(original_query)
            
            # Retrieve documents
            retrieved_results = retriever.retrieve(expanded_query, k)
            retrieved_docs = [doc_id for doc_id, _ in retrieved_results]
            
            results[query_id] = retrieved_docs
        
        # Calculate MAP
        map_score = self.evaluator.calculate_map(results)
        
        return map_score, results
    
    def run_all_experiments(self, k: int = 1000) -> pd.DataFrame:
        """Run all combinations of expansion methods and retrievers"""
        experiment_results = []
        
        print("Running IR experiments...")
        print("=" * 50)
        
        for expansion_method in self.expansion_methods:
            for retriever in self.retrievers:
                print(f"Running: {expansion_method.get_name()} + {retriever.get_name()}")
                
                map_score, detailed_results = self.run_single_experiment(
                    expansion_method, retriever, k
                )
                
                experiment_results.append({
                    'Expansion_Method': expansion_method.get_name(),
                    'Retriever': retriever.get_name(),
                    'MAP': map_score
                })
                
                print(f"MAP: {map_score:.4f}")
                print("-" * 30)
        
        # Create results DataFrame
        results_df = pd.DataFrame(experiment_results)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and display experiment results"""
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS")
        print("=" * 50)
        
        # Display all results
        print(results_df.to_string(index=False))
        
        # Best overall combination
        best_result = results_df.loc[results_df['MAP'].idxmax()]
        print(f"\nBest combination:")
        print(f"Expansion: {best_result['Expansion_Method']}")
        print(f"Retriever: {best_result['Retriever']}")
        print(f"MAP: {best_result['MAP']:.4f}")
        
        # Best per retriever
        print(f"\nBest expansion method per retriever:")
        for retriever in results_df['Retriever'].unique():
            retriever_results = results_df[results_df['Retriever'] == retriever]
            best_for_retriever = retriever_results.loc[retriever_results['MAP'].idxmax()]
            print(f"{retriever}: {best_for_retriever['Expansion_Method']} (MAP: {best_for_retriever['MAP']:.4f})")
        
        # Best per expansion method
        print(f"\nBest retriever per expansion method:")
        for expansion in results_df['Expansion_Method'].unique():
            expansion_results = results_df[results_df['Expansion_Method'] == expansion]
            best_for_expansion = expansion_results.loc[expansion_results['MAP'].idxmax()]
            print(f"{expansion}: {best_for_expansion['Retriever']} (MAP: {best_for_expansion['MAP']:.4f})")

def load_data():
    """Load your queries and relevance judgments"""
    # Replace with your data loading logic
    queries = {
        'q1': 'example query 1',
        'q2': 'example query 2',
        # Add your queries here
    }
    
    relevance_judgments = {
        'q1': {'doc1': 1, 'doc2': 0, 'doc3': 1},
        'q2': {'doc1': 0, 'doc2': 1, 'doc3': 1},
        # Add your relevance judgments here
    }
    
    return queries, relevance_judgments

def main():
    # Load data
    queries, relevance_judgments = load_data()
    
    # Create experiment
    experiment = IRExperiment(queries, relevance_judgments)
    
    # Run all experiments
    results_df = experiment.run_all_experiments(k=1000)
    
    # Analyze results
    experiment.analyze_results(results_df)
    
    # Save results
    results_df.to_csv('ir_experiment_results.csv', index=False)
    print(f"\nResults saved to 'ir_experiment_results.csv'")

if __name__ == "__main__":
    main()