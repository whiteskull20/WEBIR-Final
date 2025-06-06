import random
import mido
import os
from musiclang_predict import MusicLangPredictor
class PolyphonyMIDIExtender:
    def __init__(self, model_path='polyphony_rnn.mag',save_dir='/tmp/polyphony_rnn/generated', nb_tokens=256, temperature=1.0):
        """
        Initializes the MIDIExtender class with the given parameters.

        :param model_path: Path to the MusicLang model.
        :param nb_tokens: Number of tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Top-p sampling parameter.
        :param seed: Random seed for reproducibility.
        """
        self.model_path = model_path
        self.nb_tokens = nb_tokens
        self.temperature = temperature
        self.save_dir = save_dir

    def extend_midi(self, input_file,output_file):
        """
        Extends the given MIDI file, with optional sampling of sec seconds before extension.

        :param input_file: Path to the input MIDI file.
        :param output_file: Path to save the extended MIDI file.
        :param sec: Number of seconds to sample from the input MIDI.
        """
        os.system(f"polyphony_rnn_generate --primer_midi={input_file} --condition_on_primer=true --bundle_file={self.model_path} --output_dir={self.save_dir} --num_outputs=1 --num_steps={self.nb_tokens} --temperature={self.temperature}")
        #list all files in the save_dir
        generated_files = os.listdir(self.save_dir)
        #get the last file generated
        last_file = sorted(generated_files, key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)))[-1]
        #load the generated midi file
        #save to output_file
        generated_midi = mido.MidiFile(os.path.join(self.save_dir, last_file))
        generated_midi.save(output_file)
class MelodyRNNMIDIExtender:
    def __init__(self, model_path='basic_rnn.mag',save_dir='/tmp/melody_rnn/generated', nb_tokens=256, temperature=1.0):
        """
        Initializes the MIDIExtender class with the given parameters.

        :param model_path: Path to the MusicLang model.
        :param nb_tokens: Number of tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Top-p sampling parameter.
        :param seed: Random seed for reproducibility.
        """
        self.model_path = model_path
        self.nb_tokens = nb_tokens
        self.temperature = temperature
        self.save_dir = save_dir

    def extend_midi(self, input_file,output_file):
        """
        Extends the given MIDI file, with optional sampling of sec seconds before extension.

        :param input_file: Path to the input MIDI file.
        :param output_file: Path to save the extended MIDI file.
        :param sec: Number of seconds to sample from the input MIDI.
        """
        os.system(f"melody_rnn_generate --config={self.model_path[:-4]} --primer_midi={input_file} --condition_on_primer=true --bundle_file={self.model_path} --output_dir={self.save_dir} --num_outputs=1 --num_steps={self.nb_tokens} --temperature={self.temperature}")
        #list all files in the save_dir
        generated_files = os.listdir(self.save_dir)
        #get the last file generated
        last_file = sorted(generated_files, key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)))[-1]
        #load the generated midi file
        #save to output_file
        generated_midi = mido.MidiFile(os.path.join(self.save_dir, last_file))
        generated_midi.save(output_file)

        
class MusicLangMIDIExtender:
    def __init__(self, model_path='musiclang/musiclang-v2', nb_tokens=1024, temperature=1.0, top_p=1.0):
        """
        Initializes the MIDIExtender class with the given parameters.

        :param model_path: Path to the MusicLang model.
        :param nb_tokens: Number of tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Top-p sampling parameter.
        :param seed: Random seed for reproducibility.
        """
        self.model_path = model_path
        self.nb_tokens = nb_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.ml = MusicLangPredictor(self.model_path)

    def extend_midi(self, input_file, output_file):
        """
        Extends the given MIDI file, with optional sampling of sec seconds before extension.

        :param input_file: Path to the input MIDI file.
        :param output_file: Path to save the extended MIDI file.
        :param sec: Number of seconds to sample from the input MIDI.
        """
        
        score = self.ml.predict(
            score=input_file,
            nb_tokens=self.nb_tokens,
            temperature=self.temperature,
            topp=self.top_p,
        )
        score.to_midi(output_file)
# Example usage
if __name__ == "__main__":
    filename = "query_dataset/midi_queries/query_0437_TRBBZQM128F931A529.mid"
    extender = MusicLangMIDIExtender()
    extender.extend_midi(filename, "expanded_queries/query_0437_TRBBZQM128F931A529_1.mid")
    extender = PolyphonyMIDIExtender()
    extender.extend_midi(filename, "test_polyphony.mid")
    extender = MelodyRNNMIDIExtender()
    extender.extend_midi(filename, "test_melody_rnn.mid")
    extender = MelodyRNNMIDIExtender(model_path='attention_rnn.mag')
    extender.extend_midi(filename, "test_attention_rnn.mid")

    print("MIDI files extended successfully.")

