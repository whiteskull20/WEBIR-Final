import os
import glob
from pathlib import Path
from tqdm import tqdm
from query_expansion import MusicLangMIDIExtender, PolyphonyMIDIExtender, MelodyRNNMIDIExtender

def process_all_midi_queries():
    """Process all MIDI queries with 4 different expansion methods"""
    
    # Input and output directories
    input_dir = "query_dataset/midi_queries"
    output_dir = "query_dataset/expanded_queries"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MIDI files in the input directory
    midi_files = glob.glob(os.path.join(input_dir, "*.mid"))
    midi_files.extend(glob.glob(os.path.join(input_dir, "*.midi")))
    
    if not midi_files:
        print(f"No MIDI files found in {input_dir}")
        return
    
    print(f"Found {len(midi_files)} MIDI files to process")
    
    # Initialize extenders
    extenders = [
        ("MusicLang", MusicLangMIDIExtender()),
        ("Polyphony", PolyphonyMIDIExtender()),
        ("MelodyRNN", MelodyRNNMIDIExtender()),
        ("AttentionRNN", MelodyRNNMIDIExtender(model_path='attention_rnn.mag'))
    ]
    
    # Process each MIDI file
    total_operations = len(midi_files) * len(extenders)
    
    with tqdm(total=total_operations, desc="Processing MIDI files") as pbar:
        for midi_file in midi_files:
            # Get filename without extension
            filename = Path(midi_file).stem
            
            # Process with each extender
            for i, (method_name, extender) in enumerate(extenders, 1):
                try:
                    # Create output filename
                    output_filename = f"{filename}_{i}.mid"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing {filename} with {method_name}")
                    
                    # Extend MIDI file
                    extender.extend_midi(midi_file, output_path)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing {filename} with {method_name}: {str(e)}")
                    pbar.update(1)
                    continue
    
    print(f"\nMIDI expansion completed! Results saved in '{output_dir}' directory")
    print("Expansion methods:")
    print("  _1: MusicLang")
    print("  _2: Polyphony") 
    print("  _3: MelodyRNN")
    print("  _4: AttentionRNN")

def verify_results():
    """Verify that all expansions were created successfully"""
    input_dir = "query_dataset/midi_queries"
    output_dir = "query_dataset/expanded_queries"
    
    # Get original files
    original_files = glob.glob(os.path.join(input_dir, "*.mid"))
    original_files.extend(glob.glob(os.path.join(input_dir, "*.midi")))
    
    original_count = len(original_files)
    expected_expanded = original_count * 4
    
    # Count expanded files
    expanded_files = glob.glob(os.path.join(output_dir, "*_[1-4].mid"))
    actual_expanded = len(expanded_files)
    
    print(f"\nVerification Results:")
    print(f"Original files: {original_count}")
    print(f"Expected expanded files: {expected_expanded}")
    print(f"Actual expanded files: {actual_expanded}")
    
    if actual_expanded == expected_expanded:
        print("✓ All files processed successfully!")
    else:
        print("⚠ Some files may be missing. Check for errors above.")
        
        # Show missing files by method
        for method_num in range(1, 5):
            method_files = glob.glob(os.path.join(output_dir, f"*_{method_num}.mid"))
            method_count = len(method_files)
            if method_count != original_count:
                method_names = ["MusicLang", "Polyphony", "MelodyRNN", "AttentionRNN"]
                print(f"  Method {method_num} ({method_names[method_num-1]}): {method_count}/{original_count} files")

if __name__ == "__main__":
    try:
        process_all_midi_queries()
        verify_results()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")