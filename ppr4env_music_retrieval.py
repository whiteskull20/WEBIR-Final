#!/usr/bin/env python3
"""
PPR4ENV 音樂檢索系統實現
基於論文: "A POLYPHONIC MUSIC RETRIEVAL SYSTEM USING N-GRAMS"

此實現專注於PPR4ENV索引類型，包含位置資訊和包絡線限制
"""

import math
import itertools
import json
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

class MusicNGramExtractor:
    """專門為PPR4ENV設計的N-gram提取器"""
    
    def __init__(self):
        self.window_size = 4  # PPR4ENV固定使用n=4
        self.interval_X = 27  # 字母數量限制
        self.interval_Y = 24  # 半音映射參數
        
        # 設定節奏比值分類和邊界
        self._setup_ratio_bins_and_boundaries()
    
    def _setup_ratio_bins_and_boundaries(self):
        """設定節奏比值分類和邊界 - 根據論文修正"""
        # 論文中的峰值比值
        peak_ratios = [1.0, 6/5, 5/4, 4/3, 3/2, 5/3, 2.0, 5/2, 3.0, 4.0]
        
        # 計算峰值之間的中點作為邊界
        self.ratio_boundaries = []
        for i in range(len(peak_ratios) - 1):
            midpoint = (peak_ratios[i] + peak_ratios[i+1]) / 2
            self.ratio_boundaries.append(midpoint)
        
        # 添加邊界：0到第一個峰值的中點，和4.5作為上界
        self.ratio_boundaries.insert(0, peak_ratios[0] / 2)  # 0.5
        self.ratio_boundaries.append(4.5)  # ≥4.5編碼為Y
        
        # 編碼映射
        self.ratio_encoding = {
            1.0: 'Z',
            6/5: 'A', 5/4: 'B', 4/3: 'C', 3/2: 'D', 5/3: 'E',
            2.0: 'F', 5/2: 'G', 3.0: 'H', 4.0: 'I',
            'large': 'Y'  # ≥4.5
        }
        
        # 對應的小於1的比值編碼
        self.ratio_encoding.update({
            5/6: 'a', 4/5: 'b', 3/4: 'c', 2/3: 'd', 3/5: 'e',
            1/2: 'f', 2/5: 'g', 1/3: 'h', 1/4: 'i',
            'small': 'y'  # ≤1/4.5
        })
    
    def extract_ngrams_with_positions(self, grouped_events):
        """
        從分組事件中提取N-grams，包含位置資訊
        
        Returns:
            list: [(musical_word, onset_position), ...] 其中onset_position是窗口起始位置
        """
        if len(grouped_events) < self.window_size:
            return []
        
        all_musical_words_with_positions = []
        
        # 滑動窗口
        for window_start_idx in range(len(grouped_events) - self.window_size + 1):
            window = grouped_events[window_start_idx:window_start_idx + self.window_size]
            
            # 從窗口提取包絡線限制的路徑
            paths = self._extract_envelope_restricted_paths(window)
            
            # 為每個路徑生成音樂詞彙
            for path in paths:
                musical_word = self._generate_musical_word(path)
                if musical_word:
                    # 記錄窗口起始位置作為位置資訊
                    all_musical_words_with_positions.append((musical_word, window_start_idx))
        
        return all_musical_words_with_positions
    
    def _extract_envelope_restricted_paths(self, window):
        """提取包絡線限制的路徑"""
        paths = []
        
        # 上包絡線變化
        upper_choices = []
        for onset_time, pitches in window:
            sorted_pitches = sorted(pitches, reverse=True)
            top_2 = sorted_pitches[:2] if len(sorted_pitches) >= 2 else sorted_pitches
            upper_choices.append([(onset_time, p) for p in top_2])
        
        for combination in itertools.product(*upper_choices):
            paths.append(list(combination))
        
        # 下包絡線變化
        lower_choices = []
        for onset_time, pitches in window:
            sorted_pitches = sorted(pitches)
            bottom_2 = sorted_pitches[:2] if len(sorted_pitches) >= 2 else sorted_pitches
            lower_choices.append([(onset_time, p) for p in bottom_2])
        
        for combination in itertools.product(*lower_choices):
            paths.append(list(combination))
        
        return paths
    
    def _generate_musical_word(self, path):
        """生成音樂詞彙 - 格式: I1 R1 I2 R2 I3"""
        if len(path) != self.window_size:
            return None
        
        onset_times = [onset_time for onset_time, pitch in path]
        pitches = [pitch for onset_time, pitch in path]
        
        # 計算音程
        intervals = []
        for i in range(len(pitches) - 1):
            interval = pitches[i + 1] - pitches[i]
            intervals.append(interval)
        
        # 計算節奏比值
        ratios = []
        for i in range(len(onset_times) - 2):
            denominator = onset_times[i + 1] - onset_times[i]
            if denominator > 0:
                numerator = onset_times[i + 2] - onset_times[i + 1]
                ratio = numerator / denominator
                ratios.append(ratio)
            else:
                ratios.append(1.0)
        
        # 交替編碼
        encoded_parts = []
        for i in range(len(intervals)):
            encoded_parts.append(self._encode_interval(intervals[i]))
            if i < len(ratios):
                encoded_parts.append(self._encode_ratio(ratios[i]))
        
        return ''.join(encoded_parts)
    
    def _encode_interval(self, interval):
        """編碼音程"""
        if interval == 0:
            return '0'
        
        code = int(self.interval_X * math.tanh(interval / self.interval_Y))
        abs_code = min(abs(code), 25)
        
        if interval > 0:
            return chr(ord('A') + abs_code)
        else:
            return chr(ord('a') + abs_code)
    
    def _encode_ratio(self, ratio):
        """編碼節奏比值 - 使用邊界方法"""
        # 處理極端情況
        if ratio >= 4.5:
            return 'Y'
        elif ratio <= 1/4.5:
            return 'y'
        
        # 找到比值所在的區間
        for i in range(len(self.ratio_boundaries) - 1):
            if self.ratio_boundaries[i] <= ratio < self.ratio_boundaries[i+1]:
                # 確定對應的編碼
                if i == 0:  # 第一個區間，小於第一個中點
                    return 'f' if ratio < 1.0 else 'A'
                elif i < len(self.ratio_boundaries) - 2:
                    # 根據區間索引找到對應的峰值
                    if ratio < 1.0:
                        # 小於1的區間
                        reciprocal_ratios = [5/6, 4/5, 3/4, 2/3, 3/5, 1/2, 2/5, 1/3, 1/4]
                        if i-1 < len(reciprocal_ratios):
                            return self.ratio_encoding.get(reciprocal_ratios[i-1], 'y')
                    else:
                        # 大於1的區間
                        greater_ratios = [6/5, 5/4, 4/3, 3/2, 5/3, 2.0, 5/2, 3.0, 4.0]
                        if i-1 < len(greater_ratios):
                            return self.ratio_encoding.get(greater_ratios[i-1], 'Y')
        
        # 默認情況
        return 'Z' if abs(ratio - 1.0) < 0.1 else ('Y' if ratio > 1 else 'y')


class PPR4ENVIndexer:
    """PPR4ENV專用索引器，包含位置資訊和MODN支援 - 記憶體優化版"""
    
    def __init__(self, index_dir="ppr4env_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # 字符串池 - 避免重複存儲相同的音樂詞彙
        self.word_pool = {}  # word -> word_id
        self.id_to_word = {}  # word_id -> word
        self.next_word_id = 0
        
        # 文檔ID映射
        self.doc_pool = {}  # doc_id_str -> doc_id_int
        self.id_to_doc = {}  # doc_id_int -> doc_id_str
        self.next_doc_id = 0
        
        # 倒排索引：word_id -> [(doc_id_int, positions_array), ...]
        self.inverted_index = {}
        
        # 文檔資訊
        self.document_info = {}
        
        # 詞彙統計
        self.vocabulary = {}
        
        # 並發位置索引：使用更緊湊的結構
        # doc_id_int -> {position: [word_ids]}
        self.concurrent_words = {}
        
        self.statistics = {
            'total_documents': 0,
            'total_words': 0,
            'unique_words': 0,
            'processing_time': 0,
            'processed_files': 0
        }
    
    def _get_word_id(self, word: str) -> int:
        """獲取詞彙ID，如果不存在則創建"""
        if word not in self.word_pool:
            self.word_pool[word] = self.next_word_id
            self.id_to_word[self.next_word_id] = sys.intern(word)  # 使用字符串intern
            self.next_word_id += 1
        return self.word_pool[word]
    
    def _get_doc_id(self, doc_id_str: str) -> int:
        """獲取文檔ID，如果不存在則創建"""
        if doc_id_str not in self.doc_pool:
            self.doc_pool[doc_id_str] = self.next_doc_id
            self.id_to_doc[self.next_doc_id] = sys.intern(doc_id_str)
            self.next_doc_id += 1
        return self.doc_pool[doc_id_str]
    
    def add_document(self, doc_id: str, words_with_positions: List[Tuple[str, int]], file_info: Dict):
        """
        添加文檔到索引 - 記憶體優化版
        
        Args:
            doc_id: 文檔ID
            words_with_positions: [(word, onset_position), ...]
            file_info: 檔案資訊
        """
        if not words_with_positions:
            return
        
        # 獲取文檔ID
        doc_id_int = self._get_doc_id(doc_id)
        
        # 儲存文檔資訊
        self.document_info[doc_id_int] = {
            'file_path': sys.intern(file_info.get('file_path', '')),
            'track_id': sys.intern(file_info.get('track_id', '')),
            'file_name': sys.intern(file_info.get('file_name', '')),
            'word_count': len(words_with_positions),
            'unique_words': len(set(word for word, _ in words_with_positions))
        }
        
        # 建立倒排索引和並發位置索引
        word_positions = {}
        
        # 初始化文檔的並發詞彙索引
        if doc_id_int not in self.concurrent_words:
            self.concurrent_words[doc_id_int] = {}
        
        for word, position in words_with_positions:
            word_id = self._get_word_id(word)
            
            # 收集位置
            if word_id not in word_positions:
                word_positions[word_id] = []
            word_positions[word_id].append(position)
            
            # 記錄並發詞彙
            if position not in self.concurrent_words[doc_id_int]:
                self.concurrent_words[doc_id_int][position] = []
            if word_id not in self.concurrent_words[doc_id_int][position]:
                self.concurrent_words[doc_id_int][position].append(word_id)
            
            # 更新詞彙統計
            if word_id not in self.vocabulary:
                self.vocabulary[word_id] = {'doc_freq': 0, 'total_freq': 0}
            self.vocabulary[word_id]['total_freq'] += 1
        
        # 更新文檔頻率和倒排索引
        for word_id, positions in word_positions.items():
            self.vocabulary[word_id]['doc_freq'] += 1
            
            # 使用numpy array存儲位置以節省記憶體
            positions_array = np.array(positions, dtype=np.uint32)
            
            if word_id not in self.inverted_index:
                self.inverted_index[word_id] = []
            self.inverted_index[word_id].append((doc_id_int, positions_array))
        
        self.statistics['total_documents'] += 1
        self.statistics['total_words'] += len(words_with_positions)
        self.statistics['unique_words'] = len(self.vocabulary)
    
    def modn_search(self, query_words: List[str], window_distance: int = 3, limit: int = 10):
        """
        MODN (Musical Ordered Distance) 搜尋 - 記憶體優化版
        
        Args:
            query_words: 查詢詞彙序列
            window_distance: 允許的最大距離
            limit: 返回結果數量
            
        Returns:
            list: [(doc_id_str, score, match_details), ...]
        """
        if not query_words:
            return []
        
        # 轉換查詢詞為ID
        query_word_ids = []
        for word in query_words:
            if word in self.word_pool:
                query_word_ids.append(self.word_pool[word])
            else:
                # 查詢詞不在索引中，跳過
                continue
        
        if not query_word_ids:
            return []
        
        doc_scores = {}
        doc_match_details = {}
        
        # 對每個文檔計算MODN分數
        candidate_docs = self._get_candidate_documents(query_word_ids)
        
        for doc_id_int in candidate_docs:
            score, match_details = self._calculate_modn_score(doc_id_int, query_word_ids, query_words, window_distance)
            if score > 0:
                doc_scores[doc_id_int] = score
                doc_match_details[doc_id_int] = match_details
        
        # 排序並返回結果
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id_int, score in sorted_results[:limit]:
            doc_id_str = self.id_to_doc[doc_id_int]
            results.append((doc_id_str, score, doc_match_details[doc_id_int]))
        
        return results
    
    def _get_candidate_documents(self, query_word_ids: List[int]) -> Set[int]:
        """獲取包含任意查詢詞的候選文檔"""
        candidate_docs = set()
        for word_id in query_word_ids:
            if word_id in self.inverted_index:
                for doc_id_int, _ in self.inverted_index[word_id]:
                    candidate_docs.add(doc_id_int)
        return candidate_docs
    
    def _calculate_modn_score(self, doc_id_int: int, query_word_ids: List[int], 
                            query_words: List[str], window_distance: int) -> Tuple[float, List]:
        """
        計算MODN分數 - 基於模糊匹配點
        
        Returns:
            (score, match_details)
        """
        # 獲取文檔中所有查詢詞的位置
        word_positions = {}
        for i, word_id in enumerate(query_word_ids):
            positions = []
            if word_id in self.inverted_index:
                for d_id, pos_array in self.inverted_index[word_id]:
                    if d_id == doc_id_int:
                        positions = pos_array.tolist()
                        break
            if positions:
                word_positions[i] = set(positions)  # 使用查詢詞索引作為key
        
        if not word_positions:
            return 0.0, []
        
        # 尋找匹配序列
        total_score = 0.0
        match_details = []
        
        # 獲取所有可能的起始位置
        all_positions = set()
        for positions in word_positions.values():
            all_positions.update(positions)
        
        # 檢查每個可能的起始位置
        for start_pos in sorted(all_positions):
            sequence_score = 0
            matched_words = []
            current_pos = start_pos
            
            for i in range(len(query_word_ids)):
                if i not in word_positions:
                    continue
                
                # 在允許的窗口內尋找詞彙
                found = False
                for pos in range(current_pos, current_pos + window_distance + 1):
                    if pos in word_positions[i]:
                        sequence_score += 1
                        matched_words.append((query_words[i], pos))
                        current_pos = pos
                        found = True
                        break
                
                if not found and i == 0:
                    # 第一個詞必須找到才能開始序列
                    break
            
            if sequence_score > 0:
                total_score += sequence_score
                match_details.append({
                    'start_position': start_pos,
                    'score': sequence_score,
                    'matched_words': matched_words
                })
        
        # 考慮並發詞彙的額外加分
        concurrent_bonus = self._calculate_concurrent_bonus(doc_id_int, query_word_ids)
        total_score += concurrent_bonus
        
        return total_score, match_details
    
    def _calculate_concurrent_bonus(self, doc_id_int: int, query_word_ids: List[int]) -> float:
        """計算並發詞彙的額外加分"""
        bonus = 0.0
        query_word_id_set = set(query_word_ids)
        
        if doc_id_int in self.concurrent_words:
            for position, word_ids in self.concurrent_words[doc_id_int].items():
                concurrent_matches = set(word_ids).intersection(query_word_id_set)
                if len(concurrent_matches) > 1:
                    # 多個查詢詞在同一位置出現，給予額外加分
                    bonus += len(concurrent_matches) * 0.5
        
        return bonus
    
    def save_index(self):
        """保存索引到檔案 - 記憶體優化版"""
        # 保存ID映射
        with open(self.index_dir / "word_mappings.pkl", 'wb') as f:
            pickle.dump({
                'word_pool': self.word_pool,
                'id_to_word': self.id_to_word,
                'next_word_id': self.next_word_id
            }, f)
        
        with open(self.index_dir / "doc_mappings.pkl", 'wb') as f:
            pickle.dump({
                'doc_pool': self.doc_pool,
                'id_to_doc': self.id_to_doc,
                'next_doc_id': self.next_doc_id
            }, f)
        
        # 保存倒排索引
        with open(self.index_dir / "inverted_index.pkl", 'wb') as f:
            pickle.dump(self.inverted_index, f)
        
        # 保存並發位置索引 - 使用更緊湊的格式
        concurrent_dict = {
            doc_id: {pos: word_ids for pos, word_ids in positions.items()} 
            for doc_id, positions in self.concurrent_words.items()
        }
        with open(self.index_dir / "concurrent_words.pkl", 'wb') as f:
            pickle.dump(concurrent_dict, f)
        
        # 保存其他資訊
        with open(self.index_dir / "vocabulary.pkl", 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
        with open(self.index_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.document_info, f)
        
        with open(self.index_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2)
        
        print(f"PPR4ENV索引已保存到 {self.index_dir}")
        
        # 顯示記憶體使用統計
        self._print_memory_stats()
    
    def load_index(self):
        """從檔案載入索引 - 記憶體優化版"""
        try:
            # 載入ID映射
            with open(self.index_dir / "word_mappings.pkl", 'rb') as f:
                word_data = pickle.load(f)
                self.word_pool = word_data['word_pool']
                self.id_to_word = word_data['id_to_word']
                self.next_word_id = word_data['next_word_id']
            
            with open(self.index_dir / "doc_mappings.pkl", 'rb') as f:
                doc_data = pickle.load(f)
                self.doc_pool = doc_data['doc_pool']
                self.id_to_doc = doc_data['id_to_doc']
                self.next_doc_id = doc_data['next_doc_id']
            
            # 載入倒排索引
            with open(self.index_dir / "inverted_index.pkl", 'rb') as f:
                self.inverted_index = pickle.load(f)
            
            # 載入並發位置索引
            with open(self.index_dir / "concurrent_words.pkl", 'rb') as f:
                self.concurrent_words = pickle.load(f)
            
            # 載入其他資訊
            with open(self.index_dir / "vocabulary.pkl", 'rb') as f:
                self.vocabulary = pickle.load(f)
            
            with open(self.index_dir / "documents.pkl", 'rb') as f:
                self.document_info = pickle.load(f)
            
            with open(self.index_dir / "statistics.json", 'r', encoding='utf-8') as f:
                self.statistics = json.load(f)
            
            print(f"PPR4ENV索引已載入")
            self._print_memory_stats()
            return True
        except FileNotFoundError as e:
            print(f"索引檔案不存在: {e}")
            return False
    
    def _print_memory_stats(self):
        """顯示記憶體使用統計"""
        print(f"\n記憶體使用統計:")
        print(f"  唯一詞彙數: {len(self.word_pool)}")
        print(f"  文檔數: {len(self.doc_pool)}")
        print(f"  倒排索引大小: {len(self.inverted_index)} 個詞條")
        
        # 估算記憶體使用
        # 這只是粗略估算，實際使用可能不同
        word_memory = sys.getsizeof(self.word_pool) + sys.getsizeof(self.id_to_word)
        doc_memory = sys.getsizeof(self.doc_pool) + sys.getsizeof(self.id_to_doc)
        
        print(f"  詞彙映射記憶體: ~{word_memory / 1024 / 1024:.2f} MB")
        print(f"  文檔映射記憶體: ~{doc_memory / 1024 / 1024:.2f} MB")


class PPR4ENVSystem:
    """完整的PPR4ENV音樂檢索系統"""
    
    def __init__(self, data_path: str, index_dir: str = "ppr4env_index"):
        self.data_path = Path(data_path)
        self.index_dir = index_dir
        
        self.extractor = MusicNGramExtractor()
        self.indexer = PPR4ENVIndexer(index_dir)
    
    def build_index_from_files(self, scanner, midi_extractor, limit=None, save_progress=100, 
                          skip_count=0, resume_processed=0, resume_failed=0):
        """建立PPR4ENV索引"""
        start_time = time.time()
        
        # 掃描所有檔案
        midi_files = scanner.scan_all_midi_files()
        
        if limit:
            midi_files = midi_files[:limit]
        
        # 應用skip_count
        if skip_count > 0:
            midi_files = midi_files[skip_count:]
            # 如果有現有索引，載入它
            if self.indexer.load_index():
                print(f"載入現有索引，從第{skip_count+1}個檔案繼續")
            
            # 繼承之前的統計數量
            processed = resume_processed
            failed = resume_failed
            # 更新indexer的統計
            self.indexer.statistics['processed_files'] = skip_count
            print(f"繼承統計: 已處理={skip_count}, 成功={processed}, 失敗={failed}")
        else:
            processed = 0
            failed = 0

        print(f"開始建立PPR4ENV索引，共 {len(midi_files)} 個檔案")

        
        
        for i, (file_path, track_id, file_name) in enumerate(midi_files):
            try:
                # 提取事件
                events = midi_extractor.extract_events_from_file(file_path)
                if not events:
                    failed += 1
                    continue
                
                # 分組事件
                grouped_events = midi_extractor.group_events_by_onset(events)
                
                # 提取N-grams with positions
                words_with_positions = self.extractor.extract_ngrams_with_positions(grouped_events)
                
                if words_with_positions:
                    file_info = {
                        'file_path': file_path,
                        'track_id': track_id,
                        'file_name': file_name
                    }
                    
                    # 使用檔案名（不含副檔名）作為文檔ID
                    # doc_id = Path(file_name).stem  # 例如：1d9d16a9da90c090809c153754823c2b

                    # 或者如果需要確保唯一性，可以使用 track_id + 檔案名
                    doc_id = f"{track_id}/{Path(file_name).stem}"

                    self.indexer.add_document(doc_id, words_with_positions, file_info)
                    processed += 1
                else:
                    failed += 1
                
                # 顯示進度 - 修改顯示邏輯
                if (i + 1) % 10 == 0:
                    current_total = skip_count + i + 1
                    print(f"當前批次: {i+1}/{len(midi_files)}, 總計: {current_total}, 成功: {processed}, 失敗: {failed}")
                
                # 定期保存 - 修改checkpoint保存
                if (i + 1) % save_progress == 0:
                    current_total_processed = skip_count + i + 1
                    self.indexer.statistics['processed_files'] = current_total_processed
                    self.indexer.save_index()
                    self._save_checkpoint(current_total_processed, processed, failed)  # 修改這行
                    print(f"已保存進度: checkpoint在第{current_total_processed}個檔案")
                    
            except Exception as e:
                print(f"處理檔案 {file_path} 時出錯: {str(e)}")
                failed += 1

        # 最終保存 - 修改最終統計
        final_total_processed = skip_count + len(midi_files)
        self.indexer.statistics['processing_time'] = time.time() - start_time
        self.indexer.statistics['processed_files'] = final_total_processed
        self.indexer.save_index()
        self._save_checkpoint(final_total_processed, processed, failed)  # 修改這行

        print(f"\nPPR4ENV索引建立完成!")
        print(f"本次處理檔案: {len(midi_files)}")
        print(f"總計處理檔案: {final_total_processed}")
        print(f"總計成功檔案: {processed}")
        print(f"總計失敗檔案: {failed}")
    
    def search(self, query_events, window_distance=3, limit=10):
        """執行PPR4ENV搜尋"""
        # 分組查詢事件
        grouped_events = self._group_query_events(query_events)
        
        # 提取查詢詞彙
        query_words_with_pos = self.extractor.extract_ngrams_with_positions(grouped_events)
        query_words = [word for word, _ in query_words_with_pos]
        
        # 執行MODN搜尋
        return self.indexer.modn_search(query_words, window_distance, limit)
    
    def _group_query_events(self, events):
        """分組查詢事件"""
        if not events:
            return []
        
        grouped = defaultdict(list)
        for onset_time, pitch in events:
            grouped[onset_time].append(pitch)
        
        result = []
        for onset_time in sorted(grouped.keys()):
            pitches = sorted(grouped[onset_time])
            result.append((onset_time, pitches))
        
        return result
    
    def _save_checkpoint(self, total_processed_files, successful_files, failed_files):
        """保存checkpoint，包含成功和失敗統計"""
        import json
        checkpoint_file = Path(self.index_dir) / "checkpoint.json"
        checkpoint = {
            'total_processed_files': total_processed_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'timestamp': time.time(),
            'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self):
        """載入checkpoint"""
        import json
        checkpoint_file = Path(self.index_dir) / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None


# 使用範例
if __name__ == "__main__":
    # 這裡展示如何使用系統
    print("PPR4ENV音樂檢索系統")
    print("=" * 50)
    
    # 初始化系統
    data_path = "/path/to/lmd_matched"  # 替換為實際路徑
    system = PPR4ENVSystem(data_path)
    
    # 如果需要建立索引，使用您現有的scanner和extractor
    # from midi_file_scanner import MIDIFileScanner
    # from midi_parser import MIDIEventExtractor
    # scanner = MIDIFileScanner(data_path)
    # midi_extractor = MIDIEventExtractor()
    # system.build_index_from_files(scanner, midi_extractor, limit=100)
    
    # 載入索引並測試搜尋
    if system.indexer.load_index():
        print(f"索引載入成功")
        print(f"文檔數: {system.indexer.statistics['total_documents']}")
        print(f"詞彙數: {system.indexer.statistics['unique_words']}")
        
        # 範例查詢
        # query_events = [(0, 60), (100, 62), (200, 64), (300, 65)]  # 簡單上行音階
        # results = system.search(query_events, window_distance=3, limit=5)
        # 
        # print(f"\n搜尋結果:")
        # for doc_id, score, details in results:
        #     doc_info = system.indexer.document_info.get(doc_id, {})
        #     print(f"文檔: {doc_id} ({doc_info.get('file_name', 'N/A')})")
        #     print(f"分數: {score}")
        #     print(f"匹配詳情: {len(details)} 個匹配序列")
        #     print("-" * 30)