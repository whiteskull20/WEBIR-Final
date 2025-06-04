import os
import glob
from pathlib import Path

class MIDIFileScanner:
    def __init__(self, root_path):
        """
        初始化 MIDI 檔案掃描器
        
        Args:
            root_path: lmd_matched 資料夾的路徑
        """
        self.root_path = Path(root_path)
        
    def scan_all_midi_files(self):
        """
        掃描所有 MIDI 檔案
        
        Returns:
            list: [(檔案路徑, track_id, 檔案名), ...]
        """
        midi_files = []
        
        print(f"開始掃描 {self.root_path} 中的 MIDI 檔案...")
        
        # 使用 glob 遞歸搜尋所有 .mid 檔案
        pattern = str(self.root_path / "**" / "*.mid")
        
        for file_path in glob.glob(pattern, recursive=True):
            file_path = Path(file_path)
            
            # 提取 track_id (資料夾名稱)
            track_id = file_path.parent.name
            
            # 檢查是否符合 TR 開頭的格式
            if track_id.startswith('TR') and len(track_id) >= 5:
                midi_files.append((str(file_path), track_id, file_path.name))
        
        print(f"找到 {len(midi_files)} 個 MIDI 檔案")
        return midi_files
    
    def get_file_info(self, file_path):
        """
        獲取檔案基本資訊
        
        Args:
            file_path: MIDI 檔案路徑
            
        Returns:
            dict: 檔案資訊
        """
        file_path = Path(file_path)
        
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'track_id': file_path.parent.name,
            'file_size': file_path.stat().st_size,
            'category_path': str(file_path.parent.parent)  # E/C/C 部分
        }

# 使用範例
if __name__ == "__main__":
    # 設定你的 lmd_matched 路徑
    root_path = r"D:\desktop\College_Study\M1-2\IR\lmd_matched"
    
    scanner = MIDIFileScanner(root_path)
    
    # 掃描所有檔案
    midi_files = scanner.scan_all_midi_files()
    
    # 顯示前 10 個檔案的資訊
    for i, (file_path, track_id, file_name) in enumerate(midi_files[:10]):
        print(f"{i+1}. Track: {track_id}")
        print(f"   檔案: {file_name}")
        print(f"   路徑: {file_path}")
        print(f"   資訊: {scanner.get_file_info(file_path)}")
        print("-" * 50)