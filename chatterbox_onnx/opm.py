import os.path

from ovos_plugin_manager.templates.tts import TTS

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


if __name__ == "__main__":
    utterance = "hello world, this is chatterbox speaking!"
    tts = ChatterboxTTSPlugin()
    tts.get_tts(utterance, "test.wav")
