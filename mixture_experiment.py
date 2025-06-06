import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use a backend that works better with different environments
plt.style.use('default')
sns.set_palette("husl")

#from midi_file_scanner import MIDIFileScanner
#from midi_parser import MIDIEventExtractor
#from ppr4env_music_retrieval import PPR4ENVSystem
#from musicbert_hf.Retriever import Retriever as MusicBERTRetriever
from bm25_toolkit import BM25Retriever

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
        return query.replace('midi_queries','expanded_queries')[:-4] + '_1.mid'
    
    def get_name(self) -> str:
        return "MusicLang"

class ExpansionMethod2(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        return query.replace('midi_queries','expanded_queries')[:-4] + '_2.mid'
    
    def get_name(self) -> str:
        return "PolyphonyRNN"

class ExpansionMethod3(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        return query.replace('midi_queries','expanded_queries')[:-4] + '_3.mid'
    
    def get_name(self) -> str:
        return "MelodyRNN(basic)"

class ExpansionMethod4(QueryExpansionMethod):
    def expand_query(self, query: str) -> str:
        return query.replace('midi_queries','expanded_queries')[:-4] + '_4.mid'
    
    def get_name(self) -> str:
        return "MelogenRNN(attention)"

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

class EmptyRetriever(Retriever):
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        return [('TRCKFSL128F92CBA6E/e64c18fde99e561e087cd2722bbf42cb.mid', 0) for i in range(k)]
    
    def get_name(self) -> str:
        return "Sparse"

class SparseRetriever(Retriever):
    def __init__(self, data_path='lmd_matched', index_dir='ppr4env_index', window_distance=3):
        self.retriever = BM25Retriever('ppr4env_index', k1=1.2, b=0.75)
    
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        results = self.retriever.search(query, limit=k)
        return results
    
    def get_name(self) -> str:
        return "Sparse"

class MixMainRetriever():
    def __init__(self, resultpath='all_retrieval_results.pkl'):
        with open(resultpath, 'rb') as f:
            self.data = pickle.load(f)
    
    def query_to_query_id(self, query: str) -> str:
        queryfilename = os.path.basename(query)[:-4]  # Remove .mid extension
        subid = 0
        if 'expanded_queries' in query:
            for i in range(1, 5):
                if f'_{i}' in queryfilename:
                    subid = i
                    queryfilename = queryfilename.replace(f'_{i}', '')
                    break
        return queryfilename, subid
    
    def get_scores(self, query, time='Full', weight=0) -> List[Tuple[str, float]]:
        query_id, subid = self.query_to_query_id(query)
        keylist = ['None', 'MusicLang', 'PolyphonyRNN', 'MelodyRNN(basic)', 'MelogenRNN(attention)']
        if query_id not in self.data:
            return []
        
        original_scores = self.data[query_id]['None']['Full']
        if subid == 0:
            return original_scores
        
        scores = self.data[query_id][keylist[subid]][time]
        results = {}
        
        # Combine scores with fusion weights
        for idx, val in original_scores:
            if idx not in results:
                results[idx] = 0
            results[idx] += val * (1 - weight)
        
        for idx, val in scores:
            if idx not in results:
                results[idx] = 0
            results[idx] += val * weight
        
        return sorted(list(results.items()), key=lambda x: x[1], reverse=True)

class MixRetriever(Retriever):
    def __init__(self, mainretriever, time='Full', weight=1):
        self.mainretriever = mainretriever
        self.time = time
        self.weight = weight
    
    def retrieve(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        results = self.mainretriever.get_scores(query, self.time, self.weight)
        return results[:k]
    
    def get_name(self) -> str:
        return f"Mix(time={self.time}, weight={self.weight})"

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
        if len(sys.argv) > 1 and sys.argv[1] == 'test':
            self.retrievers = [EmptyRetriever()]
        else:
            self.mainretriever = MixMainRetriever()
            # Get available time limits from data
            sample_query = list(self.mainretriever.data.keys())[0]
            timelist = list(self.mainretriever.data[sample_query]['None'].keys())
            print(f"Available time limits: {timelist}")
            
            self.retrievers = [
                MixRetriever(self.mainretriever, time=time, weight=weight) 
                for time in timelist 
                for weight in [0, 0.05, 0.10, 0.15, 0.20, 0.25]
            ]
    
    def run_single_experiment(self, expansion_method: QueryExpansionMethod, 
                            retriever: Retriever, k: int = 1000) -> Tuple[float, Dict[str, List[str]]]:
        """Run experiment with specific expansion method and retriever"""
        results = {}
        
        for query_id, original_query in self.queries.items():
            # Expand query
            expanded_query = expansion_method.expand_query(original_query)
            
            # Retrieve documents
            try:
                retrieved_results = retriever.retrieve(expanded_query, k)
                retrieved_docs = [doc_id for doc_id, _ in retrieved_results]
            except Exception as e:
                print(f"Warning: Error retrieving for query {query_id}: {e}")
                retrieved_docs = []
            
            results[query_id] = retrieved_docs
        
        # Calculate MAP
        map_score = self.evaluator.calculate_map(results)
        
        return map_score, results
    
    def run_all_experiments(self, k: int = 1000) -> pd.DataFrame:
        """Run all combinations of expansion methods and retrievers"""
        experiment_results = []
        
        print("Running Score Fusion IR Experiments...")
        print("=" * 60)
        
        total_experiments = len(self.expansion_methods) * len(self.retrievers)
        current_experiment = 0
        
        for expansion_method in self.expansion_methods:
            for retriever in self.retrievers:
                current_experiment += 1
                print(f"[{current_experiment}/{total_experiments}] Running: {expansion_method.get_name()} + {retriever.get_name()}")
                
                map_score, detailed_results = self.run_single_experiment(
                    expansion_method, retriever, k
                )
                
                # Parse retriever parameters
                retriever_name = retriever.get_name()
                if "Mix(" in retriever_name:
                    # Extract time and weight from string like "Mix(time=Full, weight=0.5)"
                    parts = retriever_name.replace("Mix(", "").replace(")", "").split(", ")
                    time_part = parts[0].split("=")[1]
                    weight_part = float(parts[1].split("=")[1])
                else:
                    time_part = "N/A"
                    weight_part = 0.0
                
                experiment_results.append({
                    'Expansion_Method': expansion_method.get_name(),
                    'Retriever': retriever.get_name(),
                    'Time_Limit': time_part,
                    'Fusion_Weight': weight_part,
                    'MAP': map_score
                })
                
                print(f"MAP: {map_score:.4f}")
                print("-" * 40)
        
        # Create results DataFrame
        results_df = pd.DataFrame(experiment_results)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and display experiment results with enhanced insights"""
        print("\n" + "=" * 70)
        print("SCORE FUSION EXPERIMENT RESULTS")
        print("=" * 70)
        
        # Display all results
        display_df = results_df[['Expansion_Method', 'Time_Limit', 'Fusion_Weight', 'MAP']].copy()
        display_df['MAP'] = display_df['MAP'].map('{:.4f}'.format)
        print("\nAll Results:")
        print(display_df.to_string(index=False))
        
        # Best overall combination
        best_result = results_df.loc[results_df['MAP'].idxmax()]
        print(f"\n🏆 Best Overall Combination:")
        print(f"   Expansion Method: {best_result['Expansion_Method']}")
        print(f"   Time Limit: {best_result['Time_Limit']}")
        print(f"   Fusion Weight: {best_result['Fusion_Weight']}")
        print(f"   MAP: {best_result['MAP']:.4f}")
        
        # Analysis by expansion method
        print(f"\n📊 Best Results by Expansion Method:")
        for expansion in results_df['Expansion_Method'].unique():
            expansion_results = results_df[results_df['Expansion_Method'] == expansion]
            best_for_expansion = expansion_results.loc[expansion_results['MAP'].idxmax()]
            print(f"   {expansion}: MAP={best_for_expansion['MAP']:.4f} "
                  f"(Time={best_for_expansion['Time_Limit']}, Weight={best_for_expansion['Fusion_Weight']})")
        
        # Weight analysis
        print(f"\n⚖️  Fusion Weight Analysis:")
        weight_analysis = results_df.groupby('Fusion_Weight')['MAP'].agg(['mean', 'std', 'max']).round(4)
        print(weight_analysis)
        
        # Time limit analysis
        print(f"\n⏱️  Time Limit Analysis:")
        time_analysis = results_df.groupby('Time_Limit')['MAP'].agg(['mean', 'std', 'max']).round(4)
        print(time_analysis)
    
    def create_visualizations(self, results_df: pd.DataFrame, save_prefix: str = "score_fusion"):
        """Create comprehensive visualizations for score fusion analysis"""
        
        # Set up the plotting style
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # 1. Heatmap: Weight vs Time Limit (averaged across expansion methods)
        print("📊 Creating heatmap visualization...")
        plt.figure(figsize=(14, 10))
        
        # Create pivot table for heatmap
        heatmap_data = results_df.groupby(['Time_Limit', 'Fusion_Weight'])['MAP'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Time_Limit', columns='Fusion_Weight', values='MAP')
        
        plt.subplot(2, 2, 1)
        sns.heatmap(heatmap_pivot, annot=True, fmt='.4f', cmap='viridis', 
                    cbar_kws={'label': 'Mean MAP Score'})
        plt.title('Score Fusion Performance Heatmap\n(Mean MAP across all expansion methods)', fontweight='bold')
        plt.xlabel('Fusion Weight (0=Original, 1=Expanded)')
        plt.ylabel('Time Limit')
        
        # 2. Line plot: Weight effect by expansion method
        plt.subplot(2, 2, 2)
        for expansion in results_df['Expansion_Method'].unique():
            expansion_data = results_df[results_df['Expansion_Method'] == expansion]
            weight_means = expansion_data.groupby('Fusion_Weight')['MAP'].mean()
            plt.plot(weight_means.index, weight_means.values, marker='o', label=expansion, linewidth=2)
        
        plt.title('Effect of Fusion Weight by Expansion Method', fontweight='bold')
        plt.xlabel('Fusion Weight (0=Original, 1=Expanded)')
        plt.ylabel('Mean MAP Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Bar plot: Best performance by time limit
        plt.subplot(2, 2, 3)
        time_best = results_df.groupby('Time_Limit')['MAP'].max().sort_values(ascending=False)
        bars = plt.bar(range(len(time_best)), time_best.values, 
                      color=sns.color_palette("husl", len(time_best)))
        plt.title('Best MAP Score by Time Limit', fontweight='bold')
        plt.xlabel('Time Limit')
        plt.ylabel('Best MAP Score')
        plt.xticks(range(len(time_best)), time_best.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_best.values[i]:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Scatter plot: Weight vs MAP colored by time limit
        plt.subplot(2, 2, 4)
        time_limits = results_df['Time_Limit'].unique()
        colors = sns.color_palette("husl", len(time_limits))
        
        for i, time_limit in enumerate(time_limits):
            time_data = results_df[results_df['Time_Limit'] == time_limit]
            plt.scatter(time_data['Fusion_Weight'], time_data['MAP'], 
                       c=[colors[i]], label=f'{time_limit}', alpha=0.7, s=50)
        
        plt.title('Fusion Weight vs MAP Score by Time Limit', fontweight='bold')
        plt.xlabel('Fusion Weight (0=Original, 1=Expanded)')
        plt.ylabel('MAP Score')
        plt.legend(title='Time Limit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Detailed heatmap for each expansion method
        print("📈 Creating detailed expansion method heatmaps...")
        n_methods = len(results_df['Expansion_Method'].unique())
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, expansion in enumerate(results_df['Expansion_Method'].unique()):
            if i >= len(axes):
                break
                
            expansion_data = results_df[results_df['Expansion_Method'] == expansion]
            pivot_data = expansion_data.pivot(index='Time_Limit', columns='Fusion_Weight', values='MAP')
            
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='plasma',
                       ax=axes[i], cbar_kws={'label': 'MAP Score'})
            axes[i].set_title(f'{expansion} Method', fontweight='bold')
            axes[i].set_xlabel('Fusion Weight')
            axes[i].set_ylabel('Time Limit')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle('Score Fusion Performance by Expansion Method', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_by_method.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Summary statistics visualization
        print("📋 Creating summary statistics visualization...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Weight distribution analysis
        weight_stats = results_df.groupby('Fusion_Weight')['MAP'].agg(['mean', 'std'])
        ax1.errorbar(weight_stats.index, weight_stats['mean'], yerr=weight_stats['std'], 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_title('MAP Score by Fusion Weight (with std dev)', fontweight='bold')
        ax1.set_xlabel('Fusion Weight')
        ax1.set_ylabel('MAP Score')
        ax1.grid(True, alpha=0.3)
        
        # Time limit box plot
        results_df.boxplot(column='MAP', by='Time_Limit', ax=ax2)
        ax2.set_title('MAP Score Distribution by Time Limit', fontweight='bold')
        ax2.set_xlabel('Time Limit')
        ax2.set_ylabel('MAP Score')
        
        # Expansion method comparison
        expansion_means = results_df.groupby('Expansion_Method')['MAP'].mean().sort_values(ascending=True)
        bars = ax3.barh(range(len(expansion_means)), expansion_means.values,
                       color=sns.color_palette("viridis", len(expansion_means)))
        ax3.set_title('Average MAP Score by Expansion Method', fontweight='bold')
        ax3.set_xlabel('Mean MAP Score')
        ax3.set_yticks(range(len(expansion_means)))
        ax3.set_yticklabels(expansion_means.index)
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{expansion_means.values[i]:.4f}', va='center', fontweight='bold')
        
        # Fusion effectiveness analysis
        fusion_effectiveness = []
        labels = []
        
        for expansion in results_df['Expansion_Method'].unique():
            if expansion == 'None':
                continue
            expansion_data = results_df[results_df['Expansion_Method'] == expansion]
            
            # Compare pure original (weight=0) vs pure expanded (weight=1) vs best fusion
            pure_original = expansion_data[expansion_data['Fusion_Weight'] == 0]['MAP'].mean()
            pure_expanded = expansion_data[expansion_data['Fusion_Weight'] == 1]['MAP'].mean()
            best_fusion = expansion_data['MAP'].max()
            
            fusion_effectiveness.extend([pure_original, pure_expanded, best_fusion])
            labels.extend([f'{expansion}\n(Original)', f'{expansion}\n(Expanded)', f'{expansion}\n(Best Fusion)'])
        
        x_pos = np.arange(len(fusion_effectiveness))
        colors = ['lightblue', 'lightcoral', 'lightgreen'] * (len(fusion_effectiveness) // 3)
        
        bars = ax4.bar(x_pos, fusion_effectiveness, color=colors)
        ax4.set_title('Fusion Effectiveness Analysis', fontweight='bold')
        ax4.set_xlabel('Method and Strategy')
        ax4.set_ylabel('MAP Score')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{fusion_effectiveness[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ All visualizations saved with prefix '{save_prefix}'")

def get_last_two_layers(file_path):
    parent_dir = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    return os.path.join(parent_dir, filename)

def get_files_in_same_directory(file_path):
    directory = os.path.dirname(file_path)
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def load_data():
    """Load your queries and relevance judgments"""
    import json
    with open(os.path.join('query_dataset', 'queries_metadata.json'), "r") as f:
        queries = json.load(f)["queries"]
    
    tracks = {_['query_id']: _['source_file'] for _ in queries}
    queries = {_['query_id']: _['query_midi_path'] for _ in queries}
    
    relevance_judgments = {
        i: {
            get_last_two_layers(tt).replace('\\', '/')[:-4]: (1 if tt == t else 0.8) 
            for tt in get_files_in_same_directory(t)
        } 
        for i, t in tracks.items()
    }
    
    return queries, relevance_judgments

def main():
    print("🎵 Score Fusion Music IR Experiment with Enhanced Visualizations")
    print("=" * 70)
    
    # Load data
    print("📁 Loading data...")
    queries, relevance_judgments = load_data()
    print(f"   Loaded {len(queries)} queries")
    
    # Create experiment
    experiment = IRExperiment(queries, relevance_judgments)
    
    # Run all experiments
    print("\n🚀 Starting score fusion experiments...")
    results_df = experiment.run_all_experiments(k=500)
    
    # Analyze results
    experiment.analyze_results(results_df)
    
    # Create comprehensive visualizations
    print("\n🎨 Creating visualizations...")
    experiment.create_visualizations(results_df, save_prefix="score_fusion_analysis")
    
    # Save results
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'score_fusion_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    # Additional analysis: Find optimal configurations
    print("\n🔍 Optimal Configuration Analysis:")
    
    # Best weight for each expansion method
    for expansion in results_df['Expansion_Method'].unique():
        expansion_data = results_df[results_df['Expansion_Method'] == expansion]
        best_config = expansion_data.loc[expansion_data['MAP'].idxmax()]
        print(f"   {expansion}: Weight={best_config['Fusion_Weight']}, "
              f"Time={best_config['Time_Limit']}, MAP={best_config['MAP']:.4f}")
    
    print(f"\n✅ Score fusion analysis completed!")

if __name__ == "__main__":
    main()