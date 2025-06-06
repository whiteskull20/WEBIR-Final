import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

# MIDI processing
try:
    import pretty_midi
    MIDI_AVAILABLE = True
except ImportError:
    print("Warning: pretty_midi not available. MIDI duration limiting will be disabled.")
    MIDI_AVAILABLE = False

#from midi_file_scanner import MIDIFileScanner
#from midi_parser import MIDIEventExtractor
#from ppr4env_music_retrieval import PPR4ENVSystem
#from musicbert_hf.Retriever import Retriever as MusicBERTRetriever
from bm25_toolkit import BM25Retriever

class MIDIProcessor:
    """MIDI file processing utilities"""
    
    @staticmethod
    def get_midi_duration(midi_file_path: str) -> float:
        """Get MIDI file duration in seconds"""
        if not MIDI_AVAILABLE:
            return 0.0
        
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            return midi_data.get_end_time()
        except Exception as e:
            print(f"Warning: Could not get duration for {midi_file_path}: {e}")
            return 0.0
    
    @staticmethod
    def truncate_midi_file(input_path: str, output_path: str, max_duration: float) -> bool:
        """
        Truncate MIDI file to specified duration (simplified version)
        
        Args:
            input_path: Input MIDI file path
            output_path: Output MIDI file path  
            max_duration: Maximum duration in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MIDI_AVAILABLE:
            # If pretty_midi not available, just copy the file
            try:
                shutil.copy2(input_path, output_path)
                return True
            except Exception as e:
                print(f"Error copying file {input_path}: {e}")
                return False
        
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(input_path)
            
            # Get original duration
            original_duration = midi_data.get_end_time()
            
            # If file is already shorter than max_duration, just copy
            if original_duration <= max_duration:
                shutil.copy2(input_path, output_path)
                return True
            
            # Create new MIDI object with minimal setup
            truncated_midi = pretty_midi.PrettyMIDI()
            
            # Process each instrument (focus only on notes - the most important part)
            for instrument in midi_data.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name
                )
                
                # Filter notes that start before max_duration
                for note in instrument.notes:
                    if note.start < max_duration:
                        # Truncate note if it extends beyond max_duration
                        note_end = min(note.end, max_duration)
                        if note_end > note.start:  # Ensure note has positive duration
                            truncated_note = pretty_midi.Note(
                                velocity=note.velocity,
                                pitch=note.pitch,
                                start=note.start,
                                end=note_end
                            )
                            new_instrument.notes.append(truncated_note)
                
                # Safely copy control changes (optional)
                try:
                    for cc in instrument.control_changes:
                        if cc.time < max_duration:
                            new_instrument.control_changes.append(cc)
                except AttributeError:
                    pass  # Skip if no control changes
                
                # Safely copy pitch bends (optional)
                try:
                    for pb in instrument.pitch_bends:
                        if pb.time < max_duration:
                            new_instrument.pitch_bends.append(pb)
                except AttributeError:
                    pass  # Skip if no pitch bends
                
                # Add instrument if it has any notes
                if new_instrument.notes:
                    truncated_midi.instruments.append(new_instrument)
            
            # Skip metadata copying to avoid API compatibility issues
            # The core functionality (note truncation) is preserved
            
            # Save truncated MIDI
            truncated_midi.write(output_path)
            return True
            
        except Exception as e:
            print(f"Warning: Error truncating MIDI file {input_path}: {e}")
            # Fallback: copy original file
            try:
                shutil.copy2(input_path, output_path)
                print(f"Fallback: Copied original file instead of truncating")
                return True
            except Exception as copy_error:
                print(f"Error copying file {input_path}: {copy_error}")
                return False

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
    def retrieve(self, query: str, k: int = 1000, max_duration: Optional[float] = None) -> List[Tuple[str, float]]:
        """Retrieve documents for a query
        
        Args:
            query: Query file path
            k: Number of results to return
            max_duration: Maximum duration in seconds (if None, no truncation)
            
        Returns: List of (doc_id, score) tuples sorted by score desc
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class EmptyRetriever(Retriever):
    def retrieve(self, query: str, k: int = 1000, max_duration: Optional[float] = None) -> List[Tuple[str, float]]:
        return [('TRCKFSL128F92CBA6E/e64c18fde99e561e087cd2722bbf42cb.mid', 0) for i in range(k)]
    
    def get_name(self) -> str:
        return "Sparse"

class SparseRetriever(Retriever):
    def __init__(self, data_path='lmd_matched', index_dir='ppr4env_index', window_distance=3):
        self.retriever = BM25Retriever('ppr4env_index', k1=1.2, b=0.75)
        self.temp_dir = tempfile.mkdtemp(prefix="midi_truncated_")
        print(f"Created temporary directory for truncated MIDI files: {self.temp_dir}")
    
    def retrieve(self, query: str, k: int = 1000, max_duration: Optional[float] = None) -> List[Tuple[str, float]]:
        query_file = query
        
        # Apply duration truncation if specified
        if max_duration is not None:
            # Create truncated version of query file
            temp_query_file = os.path.join(self.temp_dir, f"query_{int(max_duration)}s_{os.path.basename(query)}")
            
            success = MIDIProcessor.truncate_midi_file(query, temp_query_file, max_duration)
            if success:
                query_file = temp_query_file
            else:
                print(f"Warning: Failed to truncate {query}, using original file")
        
        # Perform retrieval
        results = self.retriever.search(query_file, limit=k)
        return [(doc_id, score) for doc_id, score, details in results]
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        self.cleanup()
    
    def get_name(self) -> str:
        return "Sparse"

class ComprehensiveEvaluator:
    """Comprehensive evaluator with multiple IR metrics"""
    
    def __init__(self, relevance_judgments: Dict[str, Dict[str, int]]):
        """
        relevance_judgments: {query_id: {doc_id: relevance_score}}
        relevance_score: 0 (not relevant), 1 (relevant), or float (graded relevance)
        """
        self.relevance_judgments = relevance_judgments
    
    def calculate_precision_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if query_id not in self.relevance_judgments:
            return 0.0
        
        relevant_docs = set(doc_id for doc_id, rel in self.relevance_judgments[query_id].items() if rel > 0)
        
        if not relevant_docs:
            return 0.0
        
        # Take top k retrieved documents
        top_k_docs = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        
        return relevant_in_top_k / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if query_id not in self.relevance_judgments:
            return 0.0
        
        relevant_docs = set(doc_id for doc_id, rel in self.relevance_judgments[query_id].items() if rel > 0)
        
        if not relevant_docs:
            return 0.0
        
        # Take top k retrieved documents
        top_k_docs = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_reciprocal_rank(self, query_id: str, retrieved_docs: List[str]) -> float:
        """Calculate Reciprocal Rank for a single query"""
        if query_id not in self.relevance_judgments:
            return 0.0
        
        relevant_docs = set(doc_id for doc_id, rel in self.relevance_judgments[query_id].items() if rel > 0)
        
        if not relevant_docs:
            return 0.0
        
        # Find the rank of the first relevant document
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        
        return 0.0  # No relevant document found
    
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
    
    def calculate_ndcg_at_k(self, query_id: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        if query_id not in self.relevance_judgments:
            return 0.0
        
        relevance_scores = self.relevance_judgments[query_id]
        
        if not relevance_scores:
            return 0.0
        
        # Calculate DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            rel = relevance_scores.get(doc_id, 0)
            if rel > 0:
                dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG@K (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances) if rel > 0)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, query_id: str, retrieved_docs: List[str]) -> Dict[str, float]:
        """Comprehensive evaluation for a single query"""
        return {
            'AP': self.calculate_ap(query_id, retrieved_docs),
            'RR': self.calculate_reciprocal_rank(query_id, retrieved_docs),
            'P@5': self.calculate_precision_at_k(query_id, retrieved_docs, 5),
            'P@10': self.calculate_precision_at_k(query_id, retrieved_docs, 10),
            'P@20': self.calculate_precision_at_k(query_id, retrieved_docs, 20),
            'R@5': self.calculate_recall_at_k(query_id, retrieved_docs, 5),
            'R@10': self.calculate_recall_at_k(query_id, retrieved_docs, 10),
            'R@20': self.calculate_recall_at_k(query_id, retrieved_docs, 20),
            'NDCG@5': self.calculate_ndcg_at_k(query_id, retrieved_docs, 5),
            'NDCG@10': self.calculate_ndcg_at_k(query_id, retrieved_docs, 10),
            'NDCG@20': self.calculate_ndcg_at_k(query_id, retrieved_docs, 20)
        }
    
    def evaluate_all_queries(self, results: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate mean metrics across all queries"""
        all_metrics = defaultdict(list)
        
        for query_id, retrieved_docs in results.items():
            query_metrics = self.evaluate_query(query_id, retrieved_docs)
            for metric_name, value in query_metrics.items():
                all_metrics[metric_name].append(value)
        
        # Calculate means
        mean_metrics = {}
        for metric_name, values in all_metrics.items():
            if metric_name == 'AP':
                mean_metrics['MAP'] = np.mean(values) if values else 0.0
            elif metric_name == 'RR':
                mean_metrics['MRR'] = np.mean(values) if values else 0.0
            else:
                mean_metrics[metric_name] = np.mean(values) if values else 0.0
        
        return mean_metrics

class IRExperiment:
    def __init__(self, queries: Dict[str, str], relevance_judgments: Dict[str, Dict[str, int]], 
                 duration_limits: List[Optional[float]] = [None]):
        """
        queries: {query_id: query_text}
        relevance_judgments: {query_id: {doc_id: relevance_score}}
        duration_limits: List of duration limits in seconds (None means no limit)
        """
        self.queries = queries
        self.evaluator = ComprehensiveEvaluator(relevance_judgments)
        self.duration_limits = duration_limits
        
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
            self.sparse_retriever = EmptyRetriever()
        else:
            self.sparse_retriever = SparseRetriever()
        
        self.retrievers = [
            self.sparse_retriever,
        ]
        
        # Storage for all retrieval results: query_id -> expansion_method -> duration_limit -> [(doc_id, score), ...]
        self.all_retrieval_results = defaultdict(lambda: defaultdict(dict))
    
    def get_duration_label(self, duration: Optional[float]) -> str:
        """Get label for duration limit"""
        if duration is None:
            return "Full"
        return f"{duration}s"
    
    def run_single_experiment(self, expansion_method: QueryExpansionMethod, 
                            retriever: Retriever, duration_limit: Optional[float], 
                            k: int = 1000) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
        """Run experiment with specific expansion method, retriever, and duration limit"""
        results = {}
        retrieval_results = {}  # Store full retrieval results with scores
        
        print(f"  Processing {len(self.queries)} queries with duration limit: {self.get_duration_label(duration_limit)}")
        
        for i, (query_id, original_query) in enumerate(self.queries.items()):
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(self.queries)} queries")
            
            # Expand query
            expanded_query = expansion_method.expand_query(original_query)
            
            # Check if expanded query file exists
            if not os.path.exists(expanded_query):
                #print(f"    Warning: Query file not found: {expanded_query}, using original")
                expanded_query = original_query
            
            # Retrieve documents with duration limit
            try:
                retrieved_results = retriever.retrieve(expanded_query, k, max_duration=duration_limit)
                retrieved_docs = [doc_id for doc_id, _ in retrieved_results]
                
                # Store full retrieval results with scores
                retrieval_results[query_id] = retrieved_results
                
                # Store in all_retrieval_results for later analysis
                expansion_name = expansion_method.get_name()
                duration_label = self.get_duration_label(duration_limit)
                self.all_retrieval_results[query_id][expansion_name][duration_label] = retrieved_results
                
            except Exception as e:
                print(f"    Warning: Error retrieving for query {query_id}: {e}")
                retrieved_docs = []
                retrieval_results[query_id] = []
                
                # Store empty results
                expansion_name = expansion_method.get_name()
                duration_label = self.get_duration_label(duration_limit)
                self.all_retrieval_results[query_id][expansion_name][duration_label] = []
            
            results[query_id] = retrieved_docs
        
        # Calculate all metrics
        metrics = self.evaluator.evaluate_all_queries(results)
        
        return metrics, results, retrieval_results
    
    def analyze_query_durations(self):
        """Analyze the durations of query files"""
        print("\n🎵 Analyzing query file durations...")
        
        durations = []
        expansion_durations = defaultdict(list)
        
        for query_id, original_query in self.queries.items():
            # Check original query duration
            orig_duration = MIDIProcessor.get_midi_duration(original_query)
            durations.append(orig_duration)
            expansion_durations['Original'].append(orig_duration)
            
            # Check expanded query durations
            for expansion_method in self.expansion_methods:
                if expansion_method.get_name() == "None":
                    continue
                    
                expanded_query = expansion_method.expand_query(original_query)
                if os.path.exists(expanded_query):
                    exp_duration = MIDIProcessor.get_midi_duration(expanded_query)
                    expansion_durations[expansion_method.get_name()].append(exp_duration)
        
        # Print statistics
        if durations:
            print(f"Query duration statistics:")
            print(f"  Original queries: {len(durations)} files")
            print(f"  Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
            print(f"  Mean duration: {np.mean(durations):.2f}s")
            print(f"  Median duration: {np.median(durations):.2f}s")
            
            for expansion_name, exp_durations in expansion_durations.items():
                if exp_durations and expansion_name != 'Original':
                    print(f"  {expansion_name}: {np.mean(exp_durations):.2f}s mean")
    
    def run_all_experiments(self, k: int = 1000) -> pd.DataFrame:
        """Run all combinations of expansion methods, retrievers, and duration limits"""
        # Analyze query durations first
        self.analyze_query_durations()
        
        experiment_results = []
        
        print("\n🔍 Running comprehensive IR experiments with duration analysis...")
        print("=" * 80)
        
        total_experiments = len(self.expansion_methods) * len(self.retrievers) * len(self.duration_limits)
        current_experiment = 0
        
        for duration_limit in self.duration_limits:
            print(f"\n🕒 Duration Limit: {self.get_duration_label(duration_limit)}")
            print("=" * 60)
            
            for expansion_method in self.expansion_methods:
                for retriever in self.retrievers:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}] Running: {expansion_method.get_name()} + {retriever.get_name()}")
                    
                    metrics, detailed_results, retrieval_results = self.run_single_experiment(
                        expansion_method, retriever, duration_limit, k
                    )
                    
                    # Create result row
                    result_row = {
                        'Duration_Limit': self.get_duration_label(duration_limit),
                        'Expansion_Method': expansion_method.get_name(),
                        'Retriever': retriever.get_name(),
                    }
                    result_row.update(metrics)
                    
                    experiment_results.append(result_row)
                    
                    # Print key metrics
                    print(f"  MAP: {metrics.get('MAP', 0):.4f}")
                    print(f"  MRR: {metrics.get('MRR', 0):.4f}")
                    print(f"  P@5: {metrics.get('P@5', 0):.4f}")
                    print(f"  R@5: {metrics.get('R@5', 0):.4f}")
                    print("-" * 40)
        
        # Create results DataFrame
        results_df = pd.DataFrame(experiment_results)
        
        return results_df
    
    def save_retrieval_results(self, filename: str):
        """Save all retrieval results to pickle file"""
        import pickle
        
        # Convert defaultdict to regular dict for saving
        results_to_save = {}
        for query_id, expansion_dict in self.all_retrieval_results.items():
            results_to_save[query_id] = {}
            for expansion_name, duration_dict in expansion_dict.items():
                results_to_save[query_id][expansion_name] = dict(duration_dict)
        
        with open(filename, 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print(f"💾 Retrieval results saved to '{filename}'")
        
        # Print summary statistics
        total_combinations = 0
        total_results = 0
        for query_id, expansion_dict in results_to_save.items():
            for expansion_name, duration_dict in expansion_dict.items():
                for duration_label, results in duration_dict.items():
                    total_combinations += 1
                    total_results += len(results)
        
        print(f"📊 Summary: {len(results_to_save)} queries × "
              f"{total_combinations // len(results_to_save)} combinations = "
              f"{total_combinations} total result sets")
        print(f"📈 Total retrieval results: {total_results:,}")
        
        return filename
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyze and display experiment results with duration analysis"""
        print("\n" + "=" * 90)
        print("COMPREHENSIVE EXPERIMENT RESULTS WITH DURATION ANALYSIS")
        print("=" * 90)
        
        # Define the metrics we want to display
        metric_columns = ['MAP', 'MRR', 'P@5', 'P@10', 'P@20', 'R@5', 'R@10', 'R@20', 'NDCG@5', 'NDCG@10', 'NDCG@20']
        display_columns = ['Duration_Limit', 'Expansion_Method', 'Retriever'] + metric_columns
        
        # Display all results with formatted numbers
        display_df = results_df[display_columns].copy()
        for col in metric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].map('{:.4f}'.format)
        
        print("\nAll Results:")
        print(display_df.to_string(index=False))
        
        # Analysis by duration limit
        print(f"\n" + "=" * 70)
        print("ANALYSIS BY DURATION LIMIT")
        print("=" * 70)
        
        for duration in results_df['Duration_Limit'].unique():
            print(f"\n📏 Duration Limit: {duration}")
            duration_results = results_df[results_df['Duration_Limit'] == duration]
            
            # Best combination for this duration
            for metric in ['MAP', 'MRR', 'P@5', 'R@5']:
                if metric in duration_results.columns:
                    best_result = duration_results.loc[duration_results[metric].idxmax()]
                    print(f"  Best {metric}: {best_result[metric]:.4f} "
                          f"({best_result['Expansion_Method']} + {best_result['Retriever']})")
        
        # Analysis by expansion method across durations
        print(f"\n" + "=" * 70)
        print("EXPANSION METHOD PERFORMANCE ACROSS DURATIONS")
        print("=" * 70)
        
        for expansion in results_df['Expansion_Method'].unique():
            print(f"\n🔍 {expansion}:")
            expansion_results = results_df[results_df['Expansion_Method'] == expansion]
            
            for metric in ['MAP', 'MRR', 'P@5', 'R@5']:
                if metric in expansion_results.columns:
                    print(f"  {metric}:")
                    for duration in expansion_results['Duration_Limit'].unique():
                        duration_exp_results = expansion_results[expansion_results['Duration_Limit'] == duration]
                        if not duration_exp_results.empty:
                            avg_score = duration_exp_results[metric].mean()
                            print(f"    {duration}: {avg_score:.4f}")
        
        # Duration impact analysis
        print(f"\n" + "=" * 70)
        print("DURATION IMPACT ANALYSIS")
        print("=" * 70)
        
        # Calculate percentage change from full to shortest duration
        duration_impact = defaultdict(dict)
        
        for expansion in results_df['Expansion_Method'].unique():
            for retriever in results_df['Retriever'].unique():
                combo_results = results_df[
                    (results_df['Expansion_Method'] == expansion) & 
                    (results_df['Retriever'] == retriever)
                ]
                
                if len(combo_results) > 1:
                    full_result = combo_results[combo_results['Duration_Limit'] == 'Full']
                    
                    if not full_result.empty:
                        full_map = full_result['MAP'].iloc[0]
                        
                        for _, row in combo_results.iterrows():
                            if row['Duration_Limit'] != 'Full':
                                duration_map = row['MAP']
                                change_pct = ((duration_map - full_map) / full_map * 100) if full_map > 0 else 0
                                
                                combo_key = f"{expansion}+{retriever}"
                                duration_impact[combo_key][row['Duration_Limit']] = change_pct
        
        if duration_impact:
            print("\nMAP Score Changes (% vs Full Duration):")
            for combo, changes in duration_impact.items():
                print(f"  {combo}:")
                for duration, change in sorted(changes.items()):
                    print(f"    {duration}: {change:+.1f}%")
        
        # Statistical summary
        print(f"\n" + "=" * 70)
        print("STATISTICAL SUMMARY BY DURATION")
        print("=" * 70)
        
        for duration in results_df['Duration_Limit'].unique():
            print(f"\n{duration}:")
            duration_results = results_df[results_df['Duration_Limit'] == duration]
            summary_stats = duration_results[metric_columns].describe()
            print(summary_stats.round(4))

def get_last_two_layers(file_path):
    parent_dir = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return os.path.join(parent_dir, filename_without_ext)

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
            get_last_two_layers(tt).replace('\\', '/'): (1 if tt == t else 0.8) 
            for tt in get_files_in_same_directory(t)
        } 
        for i, t in tracks.items()
    }
    
    return queries, relevance_judgments

def main():
    print("🎵 Comprehensive Music IR Experiment Framework with Duration Analysis")
    print("=" * 80)
    
    # Load data
    print("📁 Loading data...")
    queries, relevance_judgments = load_data()
    print(f"   Loaded {len(queries)} queries")
    print(f"   Loaded relevance judgments for {len(relevance_judgments)} queries")
    
    # Define duration limits to test (in seconds)
    # None means no limit (full duration)
    duration_limits = [None, 8.0, 10.0, 12.0]
    
    print(f"\n⏱️  Duration limits to test: {[f'{d}s' if d else 'Full' for d in duration_limits]}")
    
    # Create experiment
    experiment = IRExperiment(queries, relevance_judgments, duration_limits)
    
    # Run all experiments with increased k for better fusion potential
    print("\n🚀 Starting experiments...")
    results_df = experiment.run_all_experiments(k=1000)  # Increased k for score fusion
    
    # Analyze results
    experiment.analyze_results(results_df)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'duration_analysis_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    # Save all retrieval results for score fusion analysis
    retrieval_results_file = f'all_retrieval_results_{timestamp}.pkl'
    experiment.save_retrieval_results(retrieval_results_file)
    
    # Save pivot table for easy analysis
    pivot_file = f'duration_pivot_analysis_{timestamp}.csv'
    if 'Duration_Limit' in results_df.columns:
        # Create pivot table: Duration vs (Expansion+Retriever) with MAP scores
        results_df['Method_Combo'] = results_df['Expansion_Method'] + '+' + results_df['Retriever']
        pivot_df = results_df.pivot_table(
            index='Method_Combo', 
            columns='Duration_Limit', 
            values='MAP', 
            aggfunc='mean'
        )
        pivot_df.to_csv(pivot_file)
        print(f"📊 Pivot analysis saved to '{pivot_file}'")
    
    # Save a sample of retrieval results for inspection
    sample_file = f'sample_retrieval_results_{timestamp}.json'
    try:
        import json
        sample_data = {}
        sample_queries = list(experiment.all_retrieval_results.keys())[:2]  # First 2 queries
        
        for query_id in sample_queries:
            sample_data[query_id] = {}
            for expansion_name in experiment.all_retrieval_results[query_id]:
                sample_data[query_id][expansion_name] = {}
                for duration_label in experiment.all_retrieval_results[query_id][expansion_name]:
                    # Take top 10 results for readability
                    results = experiment.all_retrieval_results[query_id][expansion_name][duration_label][:10]
                    sample_data[query_id][expansion_name][duration_label] = results
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Sample retrieval results saved to '{sample_file}' for inspection")
    except Exception as e:
        print(f"Warning: Could not save sample results: {e}")
    
    # Cleanup
    if hasattr(experiment, 'sparse_retriever'):
        experiment.sparse_retriever.cleanup()
    
    print(f"\n✅ Experiment completed! Files saved:")
    print(f"   📈 Metrics: {output_file}")
    print(f"   🗃️  Full retrieval results: {retrieval_results_file}")
    print(f"   📊 Pivot table: {pivot_file}")
    print(f"   📄 Sample results: {sample_file}")
    print(f"\n🔬 Ready for score fusion experiments!")


if __name__ == "__main__":
    main()
