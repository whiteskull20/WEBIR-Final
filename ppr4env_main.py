#!/usr/bin/env python3
"""
PPR4ENV音樂檢索系統 - 主程式 (修正版)
"""

import argparse
import sys
from pathlib import Path

# 導入必要的組件
from midi_file_scanner import MIDIFileScanner
from midi_parser import MIDIEventExtractor
from ppr4env_music_retrieval import PPR4ENVSystem

def main():
    parser = argparse.ArgumentParser(
        description="PPR4ENV音樂檢索系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    # 建立索引
    python ppr4env_main.py build --data-path /path/to/lmd_matched --limit 1000
    
    # 載入並測試索引
    python ppr4env_main.py test --data-path /path/to/lmd_matched
    
    # 搜尋（需要實作查詢介面）
    python ppr4env_main.py search --data-path /path/to/lmd_matched --query-file query.mid
        """
    )
    
    parser.add_argument('command', choices=['build', 'test', 'search'],
                       help='要執行的命令')
    parser.add_argument('--data-path', required=True,
                       help='lmd_matched 資料夾路徑')
    parser.add_argument('--index-dir', default='ppr4env_index',
                       help='索引儲存目錄 (預設: ppr4env_index)')
    parser.add_argument('--limit', type=int,
                       help='處理檔案數量限制')
    parser.add_argument('--query-file',
                       help='查詢MIDI檔案路徑 (用於search命令)')
    parser.add_argument('--window-distance', type=int, default=3,
                       help='MODN窗口距離 (預設: 3)')
    parser.add_argument('--skip', type=int, default=0,
                   help='跳過前N個檔案（用於checkpoint恢復）')
    
    args = parser.parse_args()
    
    try:
        # 初始化系統
        system = PPR4ENVSystem(args.data_path, args.index_dir)
        
        if args.command == 'build':
            # 建立索引
            print("開始建立PPR4ENV索引...")
            
            scanner = MIDIFileScanner(args.data_path)
            midi_extractor = MIDIEventExtractor()
            
            # 檢查是否有checkpoint
            skip_count = args.skip
            resume_processed = 0
            resume_failed = 0
            
            if skip_count == 0:
                # 嘗試從checkpoint自動恢復
                checkpoint_file = Path(args.index_dir) / "checkpoint.json"
                if checkpoint_file.exists():
                    try:
                        import json
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                        skip_count = checkpoint.get('total_processed_files', 0)
                        resume_processed = checkpoint.get('successful_files', 0)
                        resume_failed = checkpoint.get('failed_files', 0)
                        
                        if skip_count > 0:
                            print(f"發現checkpoint:")
                            print(f"  已處理總數: {skip_count}")
                            print(f"  成功: {resume_processed}")
                            print(f"  失敗: {resume_failed}")
                            response = input(f"是否從此處繼續？(y/n): ")
                            if response.lower() != 'y':
                                skip_count = 0
                                resume_processed = 0
                                resume_failed = 0
                    except Exception as e:
                        print(f"讀取checkpoint失敗: {e}")
                        skip_count = 0
                        resume_processed = 0
                        resume_failed = 0
            
            if skip_count > 0:
                print(f"從checkpoint恢復，跳過前{skip_count}個檔案")
                print(f"繼承統計: 成功={resume_processed}, 失敗={resume_failed}")
            
            system.build_index_from_files(
                scanner, 
                midi_extractor, 
                limit=args.limit,
                save_progress=5000,
                skip_count=skip_count,
                resume_processed=resume_processed,  # 添加這個參數
                resume_failed=resume_failed         # 添加這個參數
            )
            
        elif args.command == 'test':
            # 測試索引
            print("測試PPR4ENV索引...")
            
            if not system.indexer.load_index():
                print("錯誤: 無法載入索引，請先執行 'build' 命令")
                sys.exit(1)
            
            # 顯示索引統計
            stats = system.indexer.statistics
            print(f"\n索引統計:")
            print(f"  文檔總數: {stats['total_documents']}")
            print(f"  詞彙總數: {stats['total_words']}")
            print(f"  唯一詞彙: {stats['unique_words']}")
            print(f"  處理時間: {stats.get('processing_time', 0):.2f} 秒")
            
            # 顯示最常見的詞彙
            print(f"\n最常見的10個詞彙:")
            vocab_sorted = sorted(system.indexer.vocabulary.items(), 
                                key=lambda x: x[1]['total_freq'], 
                                reverse=True)
            for word_id, info in vocab_sorted[:10]:
                word = system.indexer.id_to_word[word_id]
                print(f"  {word}: {info['total_freq']} (出現在 {info['doc_freq']} 個文檔)")
            
            # 測試簡單查詢
            print(f"\n測試查詢功能...")
            # 使用最常見的幾個詞彙作為測試查詢
            test_word_ids = [word_id for word_id, _ in vocab_sorted[:3]]
            test_words = [system.indexer.id_to_word[word_id] for word_id in test_word_ids]
            results = system.indexer.modn_search(test_words, window_distance=3, limit=5)
            
            print(f"查詢詞彙: {test_words}")
            print(f"找到 {len(results)} 個結果:")
            for doc_id_str, score, details in results:
                # 修正: 正確獲取文檔資訊
                doc_id_int = system.indexer.doc_pool.get(doc_id_str)
                doc_info = system.indexer.document_info.get(doc_id_int, {})
                print(f"  - {doc_id_str}: 分數={score:.2f}, 檔案={doc_info.get('file_name', 'N/A')}")
            
        elif args.command == 'search':
            # 執行搜尋
            if not args.query_file:
                print("錯誤: search命令需要提供 --query-file 參數")
                sys.exit(1)
            
            print(f"搜尋查詢檔案: {args.query_file}")
            
            if not system.indexer.load_index():
                print("錯誤: 無法載入索引，請先執行 'build' 命令")
                sys.exit(1)
            
            # 處理查詢檔案
            midi_extractor = MIDIEventExtractor()
            query_events = midi_extractor.extract_events_from_file(args.query_file)
            
            if not query_events:
                print("錯誤: 無法從查詢檔案提取事件")
                sys.exit(1)
            
            print(f"查詢事件數: {len(query_events)}")
            
            # 執行搜尋  
            results = system.search(query_events, 
                                  window_distance=args.window_distance, 
                                  limit=20)
            
            print(f"\n搜尋結果 (前10個):")
            print("=" * 80)
            
            for i, (doc_id_str, score, details) in enumerate(results):
                # 修正: 正確獲取文檔資訊
                doc_id_int = system.indexer.doc_pool.get(doc_id_str)
                doc_info = system.indexer.document_info.get(doc_id_int, {})
                
                print(f"\n{i+1}. 文檔ID (Track): {doc_id_str}")
                print(f"   MIDI檔案: {doc_info.get('file_name', 'N/A')}")
                print(f"   完整路徑: {doc_info.get('file_path', 'N/A')}")
                print(f"   相似度分數: {score:.2f}")
                print(f"   匹配序列數: {len(details)}")
                
                # 顯示前幾個匹配序列的詳情
                if details:
                    print(f"   匹配詳情:")
                    for j, match in enumerate(details[:3]):  # 只顯示前3個匹配
                        print(f"     序列{j+1}: 位置={match['start_position']}, "
                              f"匹配詞數={match['score']}, "
                              f"匹配詞={len(match.get('matched_words', []))}")
                        
                        # 顯示匹配的音樂詞彙
                        if 'matched_words' in match:
                            matched_words_str = ', '.join([word for word, pos in match['matched_words'][:5]])
                            if len(match['matched_words']) > 5:
                                matched_words_str += f"... (共{len(match['matched_words'])}個)"
                            print(f"     匹配詞彙: {matched_words_str}")
                
                print("-" * 60)
        
        print("\n執行完成!")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()