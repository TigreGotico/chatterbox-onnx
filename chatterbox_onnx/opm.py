import os.path
from typing import Tuple, Dict, Any
from ovos_plugin_manager.templates.tts import TTS
from ovos_plugin_manager.templates.transformers import TTSTransformer

from chatterbox_onnx import ChatterboxOnnx


class ChatterboxTTSPlugin(TTS):

    def __init__(self, config=None):
        super().__init__(config=config)
        self.engine: ChatterboxOnnx = ChatterboxOnnx()

    def get_tts(self, sentence, wav_file, lang=None, voice=None):
        """
        Synthesize speech for a sentence and write the audio to the specified WAV file.
        
        Parameters:
            sentence (str): Text to synthesize.
            wav_file (str): Path to the output WAV file that will be written.
            lang (str, optional): Language override used to select the default voice when no `voice` is provided.
            voice (str, optional): Voice identifier override to select a specific model.
        
        Returns:
            tuple: (wav_file, phonemes) where `wav_file` is the path to the written WAV file and `phonemes` is `None` when phoneme output is not produced.
        """
        if voice is not None and voice != "default" and not os.path.isfile(voice):
            raise ValueError(
                "expected a .wav file path for reference voice")  # TODO - consider bundling some defaults and give them names
        voice = voice or self.config.get("target_voice_path")
        self.engine.synthesize(sentence,
                               target_voice_path=voice,
                               exaggeration=self.config.get("exaggeration", 0.5),
                               max_new_tokens=self.config.get("max_new_tokens", 1024),
                               output_file_name=wav_file,
                               apply_watermark=False)
        return wav_file, None




class ChatterboxTTSTransformer(TTSTransformer):
    """ runs after TTS stage but before playback"""

    def __init__(self, name="ovos-tts-transformer-chatterbox", priority=50, config=None):
        super().__init__(name, priority, config)
        self.engine: ChatterboxOnnx = ChatterboxOnnx()
        if "reference_voice" not in self.config:
            raise ValueError("Configuration must include 'reference_voice' key with path to reference WAV file")
        self.voice = self.config["reference_voice"]

    def transform(self, wav_file: str, context: dict | None = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optionally transform passed wav_file and return path to transformed file
        :param wav_file: path to wav file generated in TTS stage
        :returns: path to transformed wav file for playback
        """
        outpath = wav_file.replace(".wav", "") + "_vc.wav"
        self.engine.voice_convert(wav_file, self.voice, outpath)
        return outpath, context


if __name__ == "__main__":
    utterance = "hello world, this is chatterbox speaking!"
    tts = ChatterboxTTSPlugin()
    tts.get_tts(utterance, "test.wav")
