import os
import mido

def get_midi_length(filepath):
    """Returns the length of the MIDI file in seconds."""
    try:
        midi = mido.MidiFile(filepath)
        return midi.length
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def average_midi_length(directory):
    """Calculates the average length of MIDI files in a directory."""
    lengths = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            length = get_midi_length(os.path.join(directory, filename))
            if length is not None:
                lengths.append(length)
    
    if lengths:
        return sum(lengths) / len(lengths)
    else:
        return 0  # No valid MIDI files found

# Example usage
directory_path = "expanded_queries"
average_length = average_midi_length(directory_path)
print(f"Average MIDI length: {average_length:.2f} seconds")