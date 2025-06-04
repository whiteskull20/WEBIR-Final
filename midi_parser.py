import pretty_midi
import numpy as np
from collections import defaultdict
import warnings
import time
from midi_file_scanner import MIDIFileScanner

class MIDIEventExtractor:
    def __init__(self, error_log_file="error_files.txt"):
        """MIDI 事件提取器"""
        self.valid_files = 0
        self.invalid_files = 0
        self.error_log_file = error_log_file
        self.error_files = []  # 儲存錯誤檔案資訊
        
        # 初始化錯誤日誌檔案
        with open(self.error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"錯誤檔案記錄 - 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
    def extract_events_from_file(self, file_path):
        """
        從 MIDI 檔案提取音樂事件
        
        Args:
            file_path: MIDI 檔案路徑
            
        Returns:
            list: [(onset_time_ms, pitch), ...] 或 None (如果檔案無效)
        """
        try:
            # 載入 MIDI 檔案
            midi_data = pretty_midi.PrettyMIDI(file_path)
            
            # 收集所有音符事件
            all_events = []
            
            for instrument in midi_data.instruments:
                # 跳過打擊樂器
                if instrument.is_drum:
                    continue
                    
                for note in instrument.notes:
                    # 轉換為毫秒並取整
                    onset_time_ms = int(note.start * 1000)
                    pitch = int(note.pitch)
                    
                    all_events.append((onset_time_ms, pitch))
            
            if len(all_events) == 0:
                error_msg = "沒有有效的音符事件"
                print(f"警告: {file_path} {error_msg}")
                self._log_error_file(file_path, error_msg)
                self.invalid_files += 1
                return None
            
            # 按照 onset time 排序
            all_events.sort(key=lambda x: x[0])
            
            self.valid_files += 1
            return all_events
            
        except Exception as e:
            error_msg = str(e)
            print(f"錯誤處理檔案 {file_path}: {error_msg}")
            self._log_error_file(file_path, error_msg)
            self.invalid_files += 1
            return None
    
    def _log_error_file(self, file_path, error_msg):
        """記錄錯誤檔案到日誌"""
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 添加到錯誤列表
        self.error_files.append({
            'file_path': file_path,
            'error': error_msg,
            'timestamp': timestamp
        })
        
        # 寫入日誌檔案
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] 錯誤類型: {error_msg}\n")
            f.write(f"檔案路徑: {file_path}\n")
            f.write("-" * 50 + "\n")
    
    def group_events_by_onset(self, events):
        """
        將事件按照 onset time 分組
        
        Args:
            events: [(onset_time_ms, pitch), ...]
            
        Returns:
            list: [(onset_time_ms, [pitch1, pitch2, ...]), ...]
        """
        if not events:
            return []
            
        grouped = defaultdict(list)
        
        for onset_time, pitch in events:
            grouped[onset_time].append(pitch)
        
        # 轉換為排序的列表
        result = []
        for onset_time in sorted(grouped.keys()):
            pitches = sorted(grouped[onset_time])  # 按音高排序
            result.append((onset_time, pitches))
        
        return result
    
    def get_statistics(self):
        """取得處理統計"""
        total = self.valid_files + self.invalid_files
        return {
            'valid_files': self.valid_files,
            'invalid_files': self.invalid_files,
            'total_files': total,
            'success_rate': self.valid_files / total if total > 0 else 0,
            'error_files_count': len(self.error_files)
        }
    
    def save_error_report(self, report_file="error_report.json"):
        """保存詳細的錯誤報告"""
        import json
        
        stats = self.get_statistics()
        
        report = {
            'summary': stats,
            'error_files': self.error_files,
            'error_types': self._get_error_type_summary()
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"錯誤報告已保存到: {report_file}")
        return report_file
    
    def _get_error_type_summary(self):
        """取得錯誤類型統計"""
        from collections import Counter
        error_types = [error['error'] for error in self.error_files]
        return dict(Counter(error_types))
    
    def save_error_file_paths(self, output_file="error_file_paths.txt"):
        """保存錯誤檔案路徑列表，用於批量刪除"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 錯誤檔案路徑列表 - 總共 {len(self.error_files)} 個檔案\n")
            f.write(f"# 生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# 每行一個檔案路徑，可用於批量刪除\n")
            f.write("\n")
            
            for error_info in self.error_files:
                f.write(f"{error_info['file_path']}\n")
        
        print(f"錯誤檔案路徑列表已保存到: {output_file}")
        return output_file
    
    def finalize_error_log(self):
        """完成錯誤日誌記錄"""
        import time
        
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"處理完成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總計錯誤檔案: {len(self.error_files)}\n")
            f.write("錯誤類型統計:\n")
            
            error_types = self._get_error_type_summary()
            for error_type, count in error_types.items():
                f.write(f"  - {error_type}: {count} 個檔案\n")
        
        # 同時保存純路徑檔案和詳細報告
        self.save_error_file_paths()
        self.save_error_report()
        
        print(f"\n錯誤日誌已完成，共記錄 {len(self.error_files)} 個錯誤檔案")

# 測試範例
class MIDIProcessor:
    def __init__(self, scanner, extractor):
        self.scanner = scanner
        self.extractor = extractor
        
    def process_sample_files(self, limit=5):
        """處理樣本檔案來測試"""
        midi_files = self.scanner.scan_all_midi_files()
        
        print(f"\n開始處理前 {limit} 個 MIDI 檔案...")
        
        for i, (file_path, track_id, file_name) in enumerate(midi_files[:limit]):
            print(f"\n處理檔案 {i+1}: {track_id}/{file_name}")
            
            # 提取事件
            events = self.extractor.extract_events_from_file(file_path)
            
            if events:
                # 分組事件
                grouped_events = self.extractor.group_events_by_onset(events)
                
                print(f"  - 總事件數: {len(events)}")
                print(f"  - 不同onset時間數: {len(grouped_events)}")
                
                # 顯示前幾個事件
                print("  - 前5個事件組:")
                for j, (onset_time, pitches) in enumerate(grouped_events[:5]):
                    print(f"    {j+1}. 時間: {onset_time}ms, 音高: {pitches}")
            else:
                print("  - 檔案處理失敗")
        
        # 顯示統計
        stats = self.extractor.get_statistics()
        print(f"\n處理統計:")
        print(f"  成功: {stats['valid_files']}")
        print(f"  失敗: {stats['invalid_files']}")
        print(f"  成功率: {stats['success_rate']:.2%}")

# 使用範例
if __name__ == "__main__":
    # 設定路徑
    root_path = r"D:\desktop\College_Study\M1-2\IR\lmd_matched"
    
    # 建立處理器
    scanner = MIDIFileScanner(root_path)  # 從前面的代碼
    extractor = MIDIEventExtractor()
    processor = MIDIProcessor(scanner, extractor)
    
    # 處理樣本檔案
    processor.process_sample_files(limit=5)