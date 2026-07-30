# chatterbox-onnx

`chatterbox-onnx` is a single-file, dependency-minimal Python port of the Chatterbox speech generation model. It runs inference through ONNX Runtime, so it does not need PyTorch or other deep learning frameworks.

The package provides text-to-speech (TTS) with voice cloning and voice conversion (VC).

## Features

- **Single-file portability**: the core logic lives in one class, `ChatterboxOnnx`.
- **ONNX-only inference**: needs only `onnxruntime` and a few utility libraries, such as `librosa` for audio processing.
- **Text-to-speech (TTS)**: generate speech from text, conditioned on a reference voice.
- **Voice conversion (VC)**: convert a source speaker's voice into a target voice.
- **Quantized model option**: uses the Q4 quantized language model (`language_model_q4.onnx`) by default, which reduces the LLM component from 2 GB to 350 MB and lowers load time and memory use.
- **Batch processing**: built-in methods synthesize or convert audio across multiple reference voices and configuration settings.
- **Caching**: models download once from the Hugging Face Hub and cache in a local directory (`~/.cache/chatterbox_onnx` by default).

## Install

```bash
pip install onnxruntime librosa numpy soundfile tqdm tokenizers huggingface_hub
```

The `apply_watermark=True` option needs the separate `resemble-perth` library.

```bash
pip install resemble-perth
```

## Usage

Copy the `ChatterboxOnnx` class and the `RepetitionPenaltyLogitsProcessor` utility class into your project. When you first initialize `ChatterboxOnnx`, it downloads and caches all needed ONNX model files from the Hugging Face Hub.

### 1. Initialization

Create an instance of the synthesizer. Set `quantized=False` to use the full-precision language model. This gives a larger file size and, in some cases, higher quality.

```python
from chatterbox_onnx import ChatterboxOnnx

# Initializes the synthesizer. Models are cached in ~/.cache/chatterbox_onnx/
# Uses the smaller, quantized LLM by default.
synthesizer = ChatterboxOnnx(quantized=True)
```

### 2. Text-to-speech (TTS)

Generate audio from text by cloning a voice from a reference WAV file. If `target_voice_path` is `None`, the synthesizer downloads and uses a default reference audio.

| Parameter | Description |
| :--- | :--- |
| `text` | The input text to synthesize. |
| `target_voice_path` | Path to a WAV file of the target voice (optional). |
| `exaggeration` | Controls expressiveness, from 0.0 to 1.0. Default is 0.5. |
| `output_file_name` | The path to save the generated WAV file. |

```python
synthesizer.synthesize(
    text="The quick brown fox jumps over the lazy dog.",
    target_voice_path="path/to/your/reference_voice.wav",
    exaggeration=0.7,
    output_file_name="chatterbox_tts_output.wav",
    apply_watermark=False
)
```

### 3. Voice conversion (VC)

Convert the speech style and identity of a source audio file to match a target voice reference.

| Parameter | Description |
| :--- | :--- |
| `source_audio_path` | Path to the audio file with the speech to convert. |
| `target_voice_path` | Path to the audio file of the voice identity to clone. |
| `output_file_name` | The path to save the converted WAV file. |

```python
synthesizer.voice_convert(
    source_audio_path="path/to/source_speech.wav",
    target_voice_path="path/to/target_voice_reference.wav",
    output_file_name="converted_voice.wav",
)
```

### 4. Batch processing (TTS and VC)

#### Batch TTS example

Generate the same text across all WAV files in a folder of reference voices, and test a range of `exaggeration` values.

```python
synthesizer.batch_synthesize(
    text="This is a test of the batch synthesis function.",
    voice_folder_path="path/to/folder_of_reference_voices",
    # (start, stop, step). Tests exaggeration values 0.3, 0.4, 0.5... 1.1.
    exaggeration_range=(0.3, 1.1, 0.1),
    output_dir="batch_tts_results",
)
```

#### Batch VC example

Convert a set of source audios using a set of reference voices.

```python
synthesizer.batch_voice_convert(
    original_audios_folder="path/to/source_audios",
    voices_folder="path/to/reference_voices",
    output_dir="batch_vc_results",
    n_random=2 # For each reference voice, convert 2 random source audios
)
```

## OpenVoiceOS plugin

The package also registers as an [OpenVoiceOS](https://github.com/OpenVoiceOS) TTS plugin, through the `ovos-tts-plugin-chatterbox-onnx` entry point, and as a TTS transformer that voice-converts synthesized audio, through the `ovos-tts-transformer-chatterbox-onnx` entry point. Both implement the templates from [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager).

## Technical details

The full set of models comes from the Hugging Face Hub repository [onnx-community/chatterbox-ONNX](https://huggingface.co/onnx-community/chatterbox-ONNX).

The pipeline has four ONNX components:

1. `speech_encoder.onnx`: extracts speaker embeddings and speech tokens from a reference audio.
2. `embed_tokens.onnx`: converts text tokens into embeddings and applies the `exaggeration` feature.
3. `language_model[_q4].onnx`: the core LLM (Llama-based) that auto-regressively generates speech tokens, conditioned on text and speaker embeddings.
4. `conditional_decoder.onnx`: the vocoder that converts the sequence of generated speech tokens into a waveform.

## License

MIT
