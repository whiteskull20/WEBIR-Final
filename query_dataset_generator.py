#!/usr/bin/env python3
"""
查詢數據集生成器 - 用於PPR4ENV音樂檢索系統
從lmd_matched資料夾中生成查詢數據集，包含GROUND TRUTH
"""

import os
import random
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import time
import pretty_midi
import numpy as np

from midi_file_scanner import MIDIFileScanner
from midi_parser import MIDIEventExtractor

class QueryDatasetGenerator:
    def __init__(self, lmd_matched_path):
        """
        初始化查詢數據集生成器
        
        Args:
            lmd_matched_path: lmd_matched資料夾路徑
        """
        self.lmd_matched_path = Path(lmd_matched_path)
        self.scanner = MIDIFileScanner(lmd_matched_path)
        self.extractor = MIDIEventExtractor()
        
    def get_category_range(self, max_category):
        """
        根據輸入的三個字母生成分類範圍
        
        Args:
            max_category: 三個字母，例如 "ABC", "CCC", "ZZZ"
            
        Returns:
            list: 所有在範圍內的分類路徑
        """
        if len(max_category) != 3:
            raise ValueError("分類必須是三個字母，例如：ABC, CCC, ZZZ")
        
        max_category = max_category.upper()
        categories = []
        
        # 生成所有從AAA到max_category的組合
        for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if first > max_category[0]:
                break
                
            for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if first == max_category[0] and second > max_category[1]:
                    break
                    
                for third in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if (first == max_category[0] and second == max_category[1] and 
                        third > max_category[2]):
                        break
                    
                    category_path = f"{first}/{second}/{third}"
                    full_path = self.lmd_matched_path / first / second / third
                    
                    # 檢查路徑是否存在
                    if full_path.exists():
                        categories.append(category_path)
        
        return categories
    
    def scan_files_in_range(self, max_category):
        """
        掃描指定範圍內的所有MIDI檔案，並按TRACK分組
        
        Args:
            max_category: 最大分類範圍
            
        Returns:
            dict: {track_id: [(file_path, file_name, category), ...], ...}
        """
        categories = self.get_category_range(max_category)
        files_by_track = defaultdict(list)
        total_files = 0
        
        print(f"掃描範圍：A/A/A 到 {max_category}")
        print(f"找到 {len(categories)} 個分類")
        
        for category in categories:
            category_path = self.lmd_matched_path / category.replace('/', os.sep)
            
            if not category_path.exists():
                continue
                
            # 掃描該分類下的所有TRACK資料夾
            for track_dir in category_path.iterdir():
                if track_dir.is_dir() and track_dir.name.startswith('TR'):
                    track_id = track_dir.name
                    
                    # 在TRACK資料夾中尋找所有MIDI檔案
                    midi_files = list(track_dir.glob("*.mid"))
                    
                    if midi_files:  # 只有當有MIDI檔案時才記錄
                        for midi_file in midi_files:
                            files_by_track[track_id].append((
                                str(midi_file),
                                midi_file.name,
                                category
                            ))
                            total_files += 1
        
        print(f"總共找到 {len(files_by_track)} 個TRACK，{total_files} 個MIDI檔案")
        
        # 顯示每個TRACK的檔案數統計
        track_file_counts = {track: len(files) for track, files in files_by_track.items()}
        multi_file_tracks = {track: count for track, count in track_file_counts.items() if count > 1}
        
        if multi_file_tracks:
            print(f"發現 {len(multi_file_tracks)} 個TRACK有多個MIDI檔案：")
            for track, count in sorted(multi_file_tracks.items())[:10]:  # 只顯示前10個
                print(f"  {track}: {count} 個檔案")
            if len(multi_file_tracks) > 10:
                print(f"  ... 還有 {len(multi_file_tracks) - 10} 個")
        
        return files_by_track
    
    def extract_query_segment(self, file_path, max_duration_seconds=10):
        """
        從MIDI檔案中提取查詢片段
        
        Args:
            file_path: MIDI檔案路徑
            max_duration_seconds: 最大時長（秒）
            
        Returns:
            dict: 查詢資訊，包含事件、時長等
        """
        try:
            # 載入MIDI檔案
            midi_data = pretty_midi.PrettyMIDI(file_path)
            
            # 獲取總時長
            total_duration = midi_data.get_end_time()
            
            if total_duration <= max_duration_seconds:
                # 檔案本身就不長，直接使用整個檔案
                start_time = 0
                end_time = total_duration
            else:
                # 隨機選擇起始時間
                max_start = total_duration - max_duration_seconds
                start_time = random.uniform(0, max_start)
                end_time = start_time + max_duration_seconds
            
            # 提取指定時間範圍內的事件
            events = []
            
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                    
                for note in instrument.notes:
                    # 檢查音符是否在時間範圍內
                    if (note.start >= start_time and note.start < end_time):
                        # 轉換為相對時間（毫秒）
                        relative_onset = int((note.start - start_time) * 1000)
                        pitch = int(note.pitch)
                        events.append((relative_onset, pitch))
            
            if not events:
                return None
            
            # 按時間排序
            events.sort(key=lambda x: x[0])
            
            return {
                'events': events,
                'duration_seconds': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'original_duration': total_duration,
                'num_events': len(events)
            }
            
        except Exception as e:
            print(f"處理檔案 {file_path} 時出錯: {e}")
            return None
    
    def create_query_midi(self, events, output_path, tempo=120):
        """
        將事件轉換為查詢MIDI檔案
        
        Args:
            events: [(onset_time_ms, pitch), ...]
            output_path: 輸出MIDI檔案路徑
            tempo: 節拍速度
        """
        # 創建新的MIDI物件
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # 創建樂器
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        # 將事件轉換為音符
        for i, (onset_time_ms, pitch) in enumerate(events):
            start_time = onset_time_ms / 1000.0  # 轉換為秒
            
            # 決定音符長度（簡單策略：到下一個音符或默認長度）
            if i + 1 < len(events):
                next_onset = events[i + 1][0] / 1000.0
                duration = min(next_onset - start_time, 0.5)  # 最長0.5秒
            else:
                duration = 0.5  # 最後一個音符默認0.5秒
            
            duration = max(duration, 0.1)  # 最短0.1秒
            
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=start_time + duration
            )
            instrument.notes.append(note)
        
        midi_obj.instruments.append(instrument)
        midi_obj.write(str(output_path))
    
    def build_ground_truth(self, files_by_track, query_track_info):
        """
        建立GROUND TRUTH資料結構
        
        Args:
            files_by_track: 所有檔案按TRACK分組的字典
            query_track_info: 查詢來源的TRACK資訊
            
        Returns:
            dict: GROUND TRUTH結構
        """
        source_track = query_track_info['track_id']
        source_file = query_track_info['file_path']
        
        # GROUND TRUTH包含：
        # 1. 主要答案：來源檔案本身
        # 2. 相關答案：同一TRACK下的其他檔案（如果有的話）
        
        ground_truth = {
            'primary_answer': {
                'track_id': source_track,
                'file_path': source_file,
                'file_name': query_track_info['file_name'],
                'category': query_track_info['category'],
                'relevance_score': 1.0,  # 最高相關性
                'note': 'Source file - exact match'
            },
            'related_answers': [],
            'total_relevant_docs': 1  # 至少有來源檔案
        }
        
        # 添加同一TRACK下的其他檔案作為相關答案
        if source_track in files_by_track:
            for file_path, file_name, category in files_by_track[source_track]:
                if file_path != source_file:  # 排除來源檔案本身
                    ground_truth['related_answers'].append({
                        'track_id': source_track,
                        'file_path': file_path,
                        'file_name': file_name,
                        'category': category,
                        'relevance_score': 0.8,  # 高相關性（同一TRACK）
                        'note': 'Same track - likely related'
                    })
                    ground_truth['total_relevant_docs'] += 1
        
        return ground_truth
    
    def generate_dataset(self, max_category="CCC", num_queries=100, 
                        max_duration_seconds=10, output_dir="query_dataset"):
        """
        生成查詢數據集，包含GROUND TRUTH
        
        Args:
            max_category: 最大分類範圍（三個字母）
            num_queries: 要生成的查詢數量
            max_duration_seconds: 每個查詢的最大時長
            output_dir: 輸出目錄
        """
        print(f"開始生成查詢數據集...")
        print(f"參數設定：")
        print(f"  - 分類範圍：A/A/A 到 {max_category}")
        print(f"  - 查詢數量：{num_queries}")
        print(f"  - 最大時長：{max_duration_seconds} 秒")
        print(f"  - 輸出目錄：{output_dir}")
        
        # 創建輸出目錄
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 創建子目錄
        midi_dir = output_path / "midi_queries"
        midi_dir.mkdir(exist_ok=True)
        
        # 掃描檔案並按TRACK分組
        files_by_track = self.scan_files_in_range(max_category)
        
        if len(files_by_track) == 0:
            print("錯誤：沒有找到任何MIDI檔案")
            return
        
        # 將所有檔案展開成列表以便隨機選擇
        all_files = []
        for track_id, files in files_by_track.items():
            for file_path, file_name, category in files:
                all_files.append((file_path, track_id, file_name, category))
        
        if len(all_files) < num_queries:
            print(f"警告：找到的檔案數量 ({len(all_files)}) 少於要求的查詢數量 ({num_queries})")
            num_queries = len(all_files)
        
        # 隨機選擇檔案
        selected_files = random.sample(all_files, num_queries)
        
        queries_info = []
        ground_truth_data = []
        successful_queries = 0
        
        print(f"\n開始處理檔案...")
        
        for i, (file_path, track_id, file_name, category) in enumerate(selected_files):
            print(f"處理 {i+1}/{num_queries}: {track_id}/{file_name}")
            
            # 提取查詢片段
            query_info = self.extract_query_segment(file_path, max_duration_seconds)
            
            if query_info is None:
                print(f"  跳過：無法提取有效片段")
                continue
            
            # 生成查詢ID
            query_id = f"query_{i+1:04d}_{track_id}"
            
            # 創建查詢MIDI檔案
            query_midi_path = midi_dir / f"{query_id}.mid"
            self.create_query_midi(query_info['events'], query_midi_path)
            
            # 建立GROUND TRUTH
            query_track_info = {
                'track_id': track_id,
                'file_path': file_path,
                'file_name': file_name,
                'category': category
            }
            ground_truth = self.build_ground_truth(files_by_track, query_track_info)
            
            # 記錄查詢資訊
            query_record = {
                'query_id': query_id,
                'source_file': file_path,
                'source_track': track_id,
                'source_filename': file_name,
                'category': category,
                'query_midi_path': str(query_midi_path),
                'events': query_info['events'],
                'duration_seconds': query_info['duration_seconds'],
                'num_events': query_info['num_events'],
                'source_start_time': query_info['start_time'],
                'source_end_time': query_info['end_time'],
                'original_duration': query_info['original_duration']
            }
            
            # 記錄GROUND TRUTH
            ground_truth_record = {
                'query_id': query_id,
                'ground_truth': ground_truth
            }
            
            queries_info.append(query_record)
            ground_truth_data.append(ground_truth_record)
            successful_queries += 1
            
            print(f"  成功：{query_info['num_events']} 個事件，{query_info['duration_seconds']:.2f} 秒")
            print(f"  GT：{ground_truth['total_relevant_docs']} 個相關文檔")
        
        # 保存查詢資訊
        metadata_file = output_path / "queries_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'max_category': max_category,
                    'requested_queries': num_queries,
                    'successful_queries': successful_queries,
                    'max_duration_seconds': max_duration_seconds,
                    'total_files_scanned': len(all_files),
                    'total_tracks': len(files_by_track)
                },
                'queries': queries_info
            }, f, indent=2, ensure_ascii=False)
        
        # 保存GROUND TRUTH
        ground_truth_file = output_path / "ground_truth.json"
        with open(ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_queries': successful_queries,
                    'ground_truth_structure': {
                        'primary_answer': 'Exact source file (relevance_score = 1.0)',
                        'related_answers': 'Other files in same track (relevance_score = 0.8)',
                        'evaluation_note': 'For evaluation, primary_answer should be ranked #1'
                    }
                },
                'ground_truth': ground_truth_data
            }, f, indent=2, ensure_ascii=False)
        
        # 生成評估用的簡化GROUND TRUTH
        self.generate_evaluation_ground_truth(ground_truth_data, output_path)
        
        # 生成統計報告
        self.generate_statistics_report(queries_info, ground_truth_data, output_path, files_by_track)
        
        print(f"\n數據集生成完成！")
        print(f"成功生成 {successful_queries} 個查詢")
        print(f"輸出位置：{output_path}")
        print(f"MIDI檔案：{midi_dir}")
        print(f"查詢元數據：{metadata_file}")
        print(f"GROUND TRUTH：{ground_truth_file}")
        
        return output_path
    
    def generate_evaluation_ground_truth(self, ground_truth_data, output_path):
        """生成評估用的簡化GROUND TRUTH格式"""
        eval_gt = {}
        
        for gt_record in ground_truth_data:
            query_id = gt_record['query_id']
            gt = gt_record['ground_truth']
            
            # 簡化格式：查詢ID -> 相關文檔列表（按相關性排序）
            relevant_docs = []
            
            # 添加主要答案
            relevant_docs.append({
                'doc_id': f"{gt['primary_answer']['track_id']}/{gt['primary_answer']['file_name']}",
                'relevance_score': gt['primary_answer']['relevance_score']
            })
            
            # 添加相關答案
            for related in gt['related_answers']:
                relevant_docs.append({
                    'doc_id': f"{related['track_id']}/{related['file_name']}",
                    'relevance_score': related['relevance_score']
                })
            
            eval_gt[query_id] = relevant_docs
        
        # 保存評估用GROUND TRUTH
        eval_gt_file = output_path / "evaluation_ground_truth.json"
        with open(eval_gt_file, 'w', encoding='utf-8') as f:
            json.dump({
                'description': 'Simplified ground truth for evaluation metrics (Precision, Recall, MAP)',
                'format': 'query_id -> list of relevant documents with scores',
                'usage': 'Compare retrieval results against relevant_docs list',
                'ground_truth': eval_gt
            }, f, indent=2, ensure_ascii=False)
        
        print(f"評估用GROUND TRUTH：{eval_gt_file}")
    
    def generate_statistics_report(self, queries_info, ground_truth_data, output_path, files_by_track):
        """生成統計報告，包含GROUND TRUTH統計"""
        if not queries_info:
            return
        
        # 收集統計資訊
        durations = [q['duration_seconds'] for q in queries_info]
        event_counts = [q['num_events'] for q in queries_info]
        categories = [q['category'] for q in queries_info]
        
        # GROUND TRUTH統計
        total_relevant_docs = [gt['ground_truth']['total_relevant_docs'] for gt in ground_truth_data]
        tracks_with_multiple_files = sum(1 for track_files in files_by_track.values() if len(track_files) > 1)
        
        # 轉換NumPy數據類型為Python原生類型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 處理category_distribution的類型轉換
        unique_categories, category_counts = np.unique(categories, return_counts=True)
        category_distribution = {str(cat): int(count) for cat, count in zip(unique_categories, category_counts)}
        
        # 處理total_relevant_docs distribution的類型轉換
        unique_docs, doc_counts = np.unique(total_relevant_docs, return_counts=True)
        doc_distribution = {str(docs): int(count) for docs, count in zip(unique_docs, doc_counts)}
        
        stats = {
            'query_stats': {
                'total_queries': len(queries_info),
                'duration_stats': {
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'median': float(np.median(durations))
                },
                'event_count_stats': {
                    'mean': float(np.mean(event_counts)),
                    'std': float(np.std(event_counts)),
                    'min': int(np.min(event_counts)),
                    'max': int(np.max(event_counts)),
                    'median': float(np.median(event_counts))
                },
                'category_distribution': category_distribution
            },
            'ground_truth_stats': {
                'total_relevant_docs': {
                    'mean': float(np.mean(total_relevant_docs)),
                    'std': float(np.std(total_relevant_docs)),
                    'min': int(np.min(total_relevant_docs)),
                    'max': int(np.max(total_relevant_docs)),
                    'distribution': doc_distribution
                },
                'tracks_with_multiple_files': int(tracks_with_multiple_files),
                'total_tracks': len(files_by_track),
                'multi_file_ratio': float(tracks_with_multiple_files / len(files_by_track))
            }
        }
        
        # 保存統計報告
        stats_file = output_path / "statistics_report.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 生成可讀的統計報告
        report_file = output_path / "dataset_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("查詢數據集統計報告（含GROUND TRUTH）\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 查詢統計\n")
            f.write("-" * 30 + "\n")
            f.write(f"總查詢數量: {stats['query_stats']['total_queries']}\n\n")
            
            f.write("時長統計 (秒):\n")
            f.write(f"  平均: {stats['query_stats']['duration_stats']['mean']:.2f}\n")
            f.write(f"  標準差: {stats['query_stats']['duration_stats']['std']:.2f}\n")
            f.write(f"  最短: {stats['query_stats']['duration_stats']['min']:.2f}\n")
            f.write(f"  最長: {stats['query_stats']['duration_stats']['max']:.2f}\n")
            f.write(f"  中位數: {stats['query_stats']['duration_stats']['median']:.2f}\n\n")
            
            f.write("事件數量統計:\n")
            f.write(f"  平均: {stats['query_stats']['event_count_stats']['mean']:.1f}\n")
            f.write(f"  標準差: {stats['query_stats']['event_count_stats']['std']:.1f}\n")
            f.write(f"  最少: {stats['query_stats']['event_count_stats']['min']}\n")
            f.write(f"  最多: {stats['query_stats']['event_count_stats']['max']}\n")
            f.write(f"  中位數: {stats['query_stats']['event_count_stats']['median']:.1f}\n\n")
            
            f.write("🎯 GROUND TRUTH統計\n")
            f.write("-" * 30 + "\n")
            f.write(f"每個查詢的相關文檔數:\n")
            f.write(f"  平均: {stats['ground_truth_stats']['total_relevant_docs']['mean']:.1f}\n")
            f.write(f"  最少: {stats['ground_truth_stats']['total_relevant_docs']['min']}\n")
            f.write(f"  最多: {stats['ground_truth_stats']['total_relevant_docs']['max']}\n\n")
            
            f.write("相關文檔數分布:\n")
            for doc_count, query_count in sorted(stats['ground_truth_stats']['total_relevant_docs']['distribution'].items()):
                f.write(f"  {doc_count} 個相關文檔: {query_count} 個查詢\n")
            
            f.write(f"\nTRACK統計:\n")
            f.write(f"  總TRACK數: {stats['ground_truth_stats']['total_tracks']}\n")
            f.write(f"  有多個檔案的TRACK: {stats['ground_truth_stats']['tracks_with_multiple_files']}\n")
            f.write(f"  多檔案比例: {stats['ground_truth_stats']['multi_file_ratio']:.1%}\n\n")
            
            f.write("📁 分類分布\n")
            f.write("-" * 30 + "\n")
            for category, count in sorted(stats['query_stats']['category_distribution'].items()):
                f.write(f"  {category}: {count} 個查詢\n")
            
            f.write(f"\n💡 評估建議\n")
            f.write("-" * 30 + "\n")
            f.write("• 使用 evaluation_ground_truth.json 進行檢索評估\n")
            f.write("• 主要答案（來源檔案）應該排在第1位\n")
            f.write("• 同TRACK的其他檔案也算相關答案\n")
            f.write("• 可計算 Precision@K, Recall@K, MAP 等指標\n")
        
        print(f"統計報告已保存：{report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="PPR4ENV查詢數據集生成器（含GROUND TRUTH）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    # 生成100個查詢，範圍到CCC，最長10秒
    python query_dataset_generator.py --data-path /path/to/lmd_matched --max-category CCC --num-queries 100 --max-duration 10
    
    # 生成50個查詢，範圍到ABC，最長5秒
    python query_dataset_generator.py --data-path /path/to/lmd_matched --max-category ABC --num-queries 50 --max-duration 5

輸出檔案說明:
    • midi_queries/          - 查詢MIDI檔案
    • queries_metadata.json  - 查詢詳細資訊
    • ground_truth.json      - 完整GROUND TRUTH
    • evaluation_ground_truth.json - 評估用簡化格式
    • dataset_report.txt     - 可讀統計報告
        """
    )
    
    parser.add_argument('--data-path', required=True,
                       help='lmd_matched資料夾路徑')
    parser.add_argument('--max-category', default='CCC',
                       help='最大分類範圍（三個字母，例如：ABC, CCC, ZZZ）')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='要生成的查詢數量')
    parser.add_argument('--max-duration', type=float, default=10.0,
                       help='每個查詢的最大時長（秒）')
    parser.add_argument('--output-dir', default='query_dataset',
                       help='輸出目錄')
    parser.add_argument('--seed', type=int,
                       help='隨機種子（用於可重現的結果）')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"使用隨機種子: {args.seed}")
    
    try:
        # 初始化生成器
        generator = QueryDatasetGenerator(args.data_path)
        
        # 生成數據集
        output_path = generator.generate_dataset(
            max_category=args.max_category,
            num_queries=args.num_queries,
            max_duration_seconds=args.max_duration,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ 數據集生成完成！輸出位置：{output_path}")
        print(f"\n📋 輸出檔案：")
        print(f"  • midi_queries/ - 查詢MIDI檔案")
        print(f"  • queries_metadata.json - 查詢詳細資訊")
        print(f"  • ground_truth.json - 完整GROUND TRUTH")
        print(f"  • evaluation_ground_truth.json - 評估用格式")
        print(f"  • dataset_report.txt - 統計報告")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()