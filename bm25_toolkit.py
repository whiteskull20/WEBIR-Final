#!/usr/bin/env python3
"""
BM25 Music Retrieval Toolkit

使用範例:
    from bm25_toolkit import BM25Retriever
    
    retriever = BM25Retriever('ppr4env_index')
    results = retriever.search('query.mid', limit=10)
"""

import math
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# 導入音樂處理相關模組
try:
    from midi_parser import MIDIEventExtractor
    from ppr4env_music_retrieval import MusicNGramExtractor
except ImportError as e:
    print(f"警告: 無法導入音樂處理模組，請確保 midi_parser.py 和 ppr4env_music_retrieval.py 在路徑中: {e}")


class BM25Retriever:
    """
    BM25音樂檢索器
    專門用於PPR4ENV索引的BM25檢索
    """
    
    def __init__(self, index_dir: str, k1: float = 1.2, b: float = 0.75):
        """
        初始化BM25檢索器
        
        Args:
            index_dir: PPR4ENV索引目錄路徑
            k1: BM25參數，控制詞頻飽和點 (預設1.2)
            b: BM25參數，控制文檔長度正規化 (預設0.75)
        """
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        
        # 索引數據結構
        self.word_pool = {}          # word -> word_id
        self.id_to_word = {}         # word_id -> word
        self.doc_pool = {}           # doc_id_str -> doc_id_int
        self.id_to_doc = {}          # doc_id_int -> doc_id_str
        self.inverted_index = {}     # word_id -> [(doc_id_int, positions_array), ...]
        self.vocabulary = {}         # word_id -> {'doc_freq': int, 'total_freq': int}
        self.document_info = {}      # doc_id_int -> {'word_count': int, ...}
        
        # BM25計算所需統計
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.doc_lengths = {}        # doc_id_int -> length
        self.idf_cache = {}          # word_id -> idf_value
        
        # 音樂處理工具
        self.midi_extractor = MIDIEventExtractor() if 'MIDIEventExtractor' in globals() else None
        self.ngram_extractor = MusicNGramExtractor() if 'MusicNGramExtractor' in globals() else None
        
        # 載入索引
        self._load_index()
        self._precompute_bm25_stats()
    
    def _load_index(self):
        """載入PPR4ENV索引文件"""
        required_files = [
            'word_mappings.pkl',
            'doc_mappings.pkl', 
            'inverted_index.pkl',
            'vocabulary.pkl',
            'documents.pkl',
        ]
        
        # 檢查必要文件是否存在
        for filename in required_files:
            if not (self.index_dir / filename).exists():
                raise FileNotFoundError(f"索引文件不存在: {self.index_dir / filename}")
        
        try:
            # 載入詞彙映射
            print("載入word_mappings.pkl")
            with open(self.index_dir / 'word_mappings.pkl', 'rb') as f:
                word_data = pickle.load(f)
                self.word_pool = word_data['word_pool']
                self.id_to_word = word_data['id_to_word']
            
            print("載入doc_mappings.pkl")
            # 載入文檔映射
            with open(self.index_dir / 'doc_mappings.pkl', 'rb') as f:
                doc_data = pickle.load(f)
                self.doc_pool = doc_data['doc_pool']
                self.id_to_doc = doc_data['id_to_doc']
            
            print("載入inverted_index.pkl")
            # 載入倒排索引
            with open(self.index_dir / 'inverted_index.pkl', 'rb') as f:
                self.inverted_index = pickle.load(f)
            
            print("載入vocabulary.pkl")
            # 載入詞彙表
            with open(self.index_dir / 'vocabulary.pkl', 'rb') as f:
                self.vocabulary = pickle.load(f)
            
            print("載入documents.pkl")
            # 載入文檔信息
            with open(self.index_dir / 'documents.pkl', 'rb') as f:
                self.document_info = pickle.load(f)
    
            
            print(f"✅ BM25索引載入成功:")
            print(f"   📁 文檔數: {len(self.document_info):,}")
            print(f"   📝 詞彙數: {len(self.vocabulary):,}")
            
        except Exception as e:
            raise RuntimeError(f"載入索引時發生錯誤: {e}")
    
    def _precompute_bm25_stats(self):
        """預計算BM25所需統計信息"""
        print("🔄 預計算BM25統計信息...")
        
        # 計算文檔長度和總文檔數
        self.total_docs = len(self.document_info)
        total_length = 0
        
        for doc_id_int, doc_info in self.document_info.items():
            doc_length = doc_info.get('word_count', 0)
            self.doc_lengths[doc_id_int] = doc_length
            total_length += doc_length
        
        # 計算平均文檔長度
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        # 預計算所有詞彙的IDF值
        for word_id, vocab_info in self.vocabulary.items():
            doc_freq = vocab_info.get('doc_freq', 0)
            if doc_freq > 0:
                # BM25 IDF公式: log((N - df + 0.5) / (df + 0.5))
                idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                self.idf_cache[word_id] = max(0.0, idf)  # 確保IDF非負
            else:
                self.idf_cache[word_id] = 0.0
        
        print(f"✅ BM25統計完成:")
        print(f"   📊 總文檔數: {self.total_docs:,}")
        print(f"   📏 平均文檔長度: {self.avg_doc_length:.2f}")
        print(f"   🔤 IDF值範圍: {min(self.idf_cache.values()):.3f} - {max(self.idf_cache.values()):.3f}")
    
    def search(self, query_midi_file: str, limit: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        從MIDI文件執行BM25搜尋
        
        Args:
            query_midi_file: 查詢MIDI文件路徑
            limit: 返回結果數量限制
            
        Returns:
            List[Tuple[str, float, Dict]]: [(doc_id, bm25_score, details), ...]
        """
        if not self.midi_extractor or not self.ngram_extractor:
            raise RuntimeError("音樂處理模組未正確載入，請檢查依賴")
        
        print(f"🎵 處理查詢文件: {query_midi_file}")
        
        # 1. 提取MIDI事件
        events = self.midi_extractor.extract_events_from_file(query_midi_file)
        if not events:
            raise ValueError(f"無法從 {query_midi_file} 提取音樂事件")
        
        # 2. 分組事件
        grouped_events = self._group_events_by_onset(events)
        if not grouped_events:
            raise ValueError("無法分組音樂事件")
        
        # 3. 提取N-grams
        words_with_positions = self.ngram_extractor.extract_ngrams_with_positions(grouped_events)
        if not words_with_positions:
            raise ValueError("無法生成音樂詞彙")
        
        query_words = [word for word, _ in words_with_positions]
        
        #print(f"📝 生成查詢詞彙: {len(query_words)} 個")
        #print(f"   前10個: {query_words[:10]}")
        
        # 4. 執行BM25搜尋
        return self.search_words(query_words, limit)
    
    def search_words(self, query_words: List[str], limit: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        使用音樂詞彙執行BM25搜尋
        
        Args:
            query_words: 查詢詞彙列表
            limit: 返回結果數量限制
            
        Returns:
            List[Tuple[str, float, Dict]]: [(doc_id, bm25_score, details), ...]
        """
        if not query_words:
            return []
        
        #print(f"🔍 執行BM25搜尋，查詢詞彙: {len(query_words)} 個")
        
        # 轉換查詢詞為word_id
        query_word_ids = []
        query_word_mapping = {}  # word_id -> original_word
        
        for word in query_words:
            if word in self.word_pool:
                word_id = self.word_pool[word]
                query_word_ids.append(word_id)
                query_word_mapping[word_id] = word
        
        if not query_word_ids:
            print("❌ 查詢詞彙在索引中未找到")
            return []
        
        #print(f"📊 有效查詢詞彙: {len(query_word_ids)}/{len(query_words)}")
        
        # 獲取候選文檔
        candidate_docs = self._get_candidate_documents(query_word_ids)
        #print(f"🎯 候選文檔數: {len(candidate_docs)}")
        
        if not candidate_docs:
            return []
        
        # 計算BM25分數
        doc_scores = defaultdict(float)
        doc_details = defaultdict(lambda: {
            'matched_terms': {},
            'doc_length': 0,
            'query_terms_found': 0,
            'total_query_terms': len(query_word_ids),
            'coverage': 0.0
        })
        
        # 對每個查詢詞計算其對所有候選文檔的貢獻
        for word_id in query_word_ids:
            if word_id not in self.inverted_index:
                continue
            
            word = query_word_mapping[word_id]
            idf = self.idf_cache.get(word_id, 0.0)
            
            # 遍歷包含此詞的所有文檔
            for doc_id_int, positions_array in self.inverted_index[word_id]:
                if doc_id_int not in candidate_docs:
                    continue
                
                # 計算詞頻(TF)
                tf = len(positions_array)
                
                # 獲取文檔長度
                doc_length = self.doc_lengths.get(doc_id_int, 1)
                
                # BM25分數計算
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                term_score = idf * (numerator / denominator)
                
                # 累加分數
                doc_scores[doc_id_int] += term_score
                
                # 記錄詳細信息
                doc_details[doc_id_int]['matched_terms'][word] = {
                    'tf': tf,
                    'idf': idf,
                    'term_score': term_score,
                    'positions_count': len(positions_array)
                }
                doc_details[doc_id_int]['doc_length'] = doc_length
                doc_details[doc_id_int]['query_terms_found'] += 1
        
        # 計算覆蓋率
        for doc_id_int in doc_details:
            found = doc_details[doc_id_int]['query_terms_found']
            total = doc_details[doc_id_int]['total_query_terms']
            doc_details[doc_id_int]['coverage'] = found / total if total > 0 else 0.0
        
        # 排序結果
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 格式化輸出
        results = []
        for doc_id_int, score in sorted_results[:limit]:
            doc_id_str = self.id_to_doc[doc_id_int]
            details = dict(doc_details[doc_id_int])
            results.append((doc_id_str, score, details))
        
        #print(f"✅ 找到 {len(results)} 個結果")
        return results
    
    def _get_candidate_documents(self, query_word_ids: List[int]) -> set:
        """獲取包含任意查詢詞的候選文檔"""
        candidate_docs = set()
        for word_id in query_word_ids:
            if word_id in self.inverted_index:
                for doc_id_int, _ in self.inverted_index[word_id]:
                    candidate_docs.add(doc_id_int)
        return candidate_docs
    
    def _group_events_by_onset(self, events: List[Tuple[int, int]]) -> List[Tuple[int, List[int]]]:
        """將音樂事件按onset時間分組"""
        if not events:
            return []
        
        grouped = defaultdict(list)
        for onset_time, pitch in events:
            grouped[onset_time].append(pitch)
        
        # 轉換為排序列表
        result = []
        for onset_time in sorted(grouped.keys()):
            pitches = sorted(grouped[onset_time])
            result.append((onset_time, pitches))
        
        return result
    
    def explain_score(self, doc_id: str, query_words: List[str]) -> Dict:
        """
        解釋特定文檔的BM25分數計算
        
        Args:
            doc_id: 文檔ID
            query_words: 查詢詞彙列表
            
        Returns:
            Dict: 詳細的分數解釋
        """
        if doc_id not in self.doc_pool:
            return {"error": f"文檔不存在: {doc_id}"}
        
        doc_id_int = self.doc_pool[doc_id]
        doc_length = self.doc_lengths.get(doc_id_int, 0)
        
        explanation = {
            'doc_id': doc_id,
            'doc_length': doc_length,
            'avg_doc_length': self.avg_doc_length,
            'total_docs': self.total_docs,
            'bm25_params': {'k1': self.k1, 'b': self.b},
            'term_explanations': [],
            'total_score': 0.0
        }
        
        total_score = 0.0
        
        for word in query_words:
            if word not in self.word_pool:
                continue
            
            word_id = self.word_pool[word]
            
            # 檢查詞是否在文檔中
            tf = 0
            positions = []
            
            if word_id in self.inverted_index:
                for d_id, pos_array in self.inverted_index[word_id]:
                    if d_id == doc_id_int:
                        tf = len(pos_array)
                        positions = pos_array.tolist() if hasattr(pos_array, 'tolist') else list(pos_array)
                        break
            
            # 計算分數組件
            idf = self.idf_cache.get(word_id, 0.0)
            doc_freq = self.vocabulary.get(word_id, {}).get('doc_freq', 0)
            
            if tf > 0:
                # BM25組件計算
                numerator = tf * (self.k1 + 1)
                length_norm = 1 - self.b + self.b * doc_length / self.avg_doc_length
                denominator = tf + self.k1 * length_norm
                term_score = idf * (numerator / denominator)
                total_score += term_score
            else:
                term_score = 0.0
            
            term_explanation = {
                'term': word,
                'tf': tf,
                'doc_freq': doc_freq,
                'idf': idf,
                'idf_calculation': f"log(({self.total_docs} - {doc_freq} + 0.5) / ({doc_freq} + 0.5))",
                'bm25_numerator': tf * (self.k1 + 1) if tf > 0 else 0,
                'length_normalization': 1 - self.b + self.b * doc_length / self.avg_doc_length,
                'bm25_denominator': tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length) if tf > 0 else 0,
                'term_score': term_score,
                'positions': positions[:20] if len(positions) <= 20 else positions[:20] + ['...']  # 最多顯示20個位置
            }
            
            explanation['term_explanations'].append(term_explanation)
        
        explanation['total_score'] = total_score
        return explanation
    
    def get_doc_info(self, doc_id: str) -> Dict:
        """
        獲取文檔詳細信息
        
        Args:
            doc_id: 文檔ID
            
        Returns:
            Dict: 文檔信息
        """
        if doc_id not in self.doc_pool:
            return {"error": f"文檔不存在: {doc_id}"}
        
        doc_id_int = self.doc_pool[doc_id]
        doc_info = self.document_info.get(doc_id_int, {})
        
        return {
            'doc_id': doc_id,
            'file_name': doc_info.get('file_name', 'N/A'),
            'file_path': doc_info.get('file_path', 'N/A'),
            'word_count': doc_info.get('word_count', 0),
            'unique_words': doc_info.get('unique_words', 0),
            'track_id': doc_info.get('track_id', 'N/A')
        }
    
    def get_statistics(self) -> Dict:
        """
        獲取BM25檢索器統計信息
        
        Returns:
            Dict: 統計信息
        """
        idf_values = list(self.idf_cache.values())
        doc_lengths = list(self.doc_lengths.values())
        
        return {
            'index_stats': {
                'total_documents': self.total_docs,
                'total_vocabulary': len(self.vocabulary),
                'avg_doc_length': self.avg_doc_length
            },
            'bm25_params': {
                'k1': self.k1,
                'b': self.b
            },
            'doc_length_stats': {
                'min': min(doc_lengths) if doc_lengths else 0,
                'max': max(doc_lengths) if doc_lengths else 0,
                'mean': np.mean(doc_lengths) if doc_lengths else 0,
                'std': np.std(doc_lengths) if doc_lengths else 0
            },
            'idf_stats': {
                'min': min(idf_values) if idf_values else 0,
                'max': max(idf_values) if idf_values else 0,
                'mean': np.mean(idf_values) if idf_values else 0,
                'std': np.std(idf_values) if idf_values else 0
            }
        }


# 工具函數
def display_results(results: List[Tuple[str, float, Dict]], query_description: str = ""):
    """
    友好地顯示BM25檢索結果
    
    Args:
        results: BM25檢索結果
        query_description: 查詢描述
    """
    print(f"\n🎯 BM25檢索結果 {f'({query_description})' if query_description else ''}")
    print(f"找到 {len(results)} 個結果:")
    print("=" * 80)
    
    if not results:
        print("❌ 沒有找到匹配結果")
        return
    
    for i, (doc_id, score, details) in enumerate(results):
        print(f"\n{i+1:2d}. 📄 文檔: {doc_id}")
        print(f"     📊 BM25分數: {score:.4f}")
        print(f"     🎯 匹配詞彙: {details['query_terms_found']}/{details['total_query_terms']}")
        print(f"     📈 覆蓋率: {details['coverage']:.2%}")
        print(f"     📏 文檔長度: {details['doc_length']}")
        
        # 顯示匹配的詞彙詳情
        if details['matched_terms']:
            matched_terms = list(details['matched_terms'].keys())[:5]  # 只顯示前5個
            print(f"     🔤 匹配詞彙: {', '.join(matched_terms)}")
            if len(details['matched_terms']) > 5:
                print(f"          ... 還有 {len(details['matched_terms']) - 5} 個詞彙")


# 使用範例
def demo_usage():
    """示範如何使用BM25檢索器"""
    print("🔍 BM25 Music Retrieval Toolkit 示範")
    print("=" * 50)
    
    try:
        # 初始化檢索器
        retriever = BM25Retriever('ppr4env_index', k1=1.2, b=0.75)
        
        print("\n📝 示範:")
        results = retriever.search(r'.\lmd_matched\A\A\D\TRAADKW128E079503A\72cae5077339f6abaee4cad318b1e923.mid ', limit=20)
        display_results(results, "MIDI文件查詢")
        results = retriever.search(r'.\lmd_matched\A\B\Z\TRABZDG128F931803F\d6371159fa42356ebd38380f7f0feec2.mid', limit=20)
        display_results(results, "MIDI文件查詢")
        results = retriever.search(r'.\lmd_matched\A\B\R\TRABRUJ128F42ADE0D\025fb4d42ff236cb25881dde5c380c1a.mid', limit=20)
        display_results(results, "MIDI文件查詢")
        results = retriever.search(r'.\lmd_matched\A\C\Q\TRACQIB128F147DAC4\313b6893e29a54a5872df5c92199eadf.mid', limit=20)
        display_results(results, "MIDI文件查詢")
        

    except Exception as e:
        print(f"❌ 示範執行失敗: {e}")


if __name__ == "__main__":
    demo_usage()