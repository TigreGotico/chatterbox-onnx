# 🤖 chatterbox-onnx: Standalone ONNX-Only Speech Synthesis

**`chatterbox-onnx`** is a **single-file, dependency-minimal Python port** of the Chatterbox speech generation model. It leverages **ONNX Runtime** for all inference, eliminating the need for PyTorch or other complex deep learning frameworks for deployment.

This solution provides high-quality **Text-to-Speech (TTS)** and **Voice Conversion (VC)** capabilities with minimal setup.

## ✨ Features

  * **Single File Portability**: The entire core logic is contained within one Python file/class (`ChatterboxOnnx`).
  * **ONNX-Only Inference**: Requires only `onnxruntime` and essential utility libraries (like `librosa` for audio processing).
  * **Text-to-Speech (TTS)**: Generate speech from text, conditioned on a reference voice (voice cloning).
  * **Voice Conversion (VC)**: Convert one person's speaking voice (source) into another person's voice (target reference).
  * **Quantized Model Option**: Uses the **Q4 quantized Language Model** (`language_model_q4.onnx`) by default, reducing the LLM component size from 2GB to 350MB for faster loading and lower memory usage.
  * **Batch Processing**: Built-in methods for synthesizing or converting audio across multiple reference voices and configuration settings.
  * **Caching**: Models are automatically downloaded and cached from the Hugging Face Hub into a local directory (`~/.cache/chatterbox_onnx` by default).

-----

## 💻 Prerequisites

To use this file, ensure you have the required Python packages installed.

```bash
pip install onnxruntime librosa numpy soundfile tqdm tokenizers huggingface_hub
```

**Note on Watermarking (Optional):**
The `apply_watermark=True` feature requires the separate installation of the `resemble-perth` library:

```bash
pip install resemble-perth
```

-----

## 🛠️ Usage

Simply copy the `ChatterboxOnnx` class and the `RepetitionPenaltyLogitsProcessor` utility class into your project. The first time you initialize the `ChatterboxOnnx` class, it will automatically download and cache all necessary ONNX model files from the Hugging Face Hub.

### 1\. Initialization

Create an instance of the synthesizer. Use `quantized=False` to use the full-precision Language Model (larger file size, potentially higher quality).

```python
from chatterbox_onnx import ChatterboxOnnx

# Initializes the synthesizer. Models will be cached in ~/.cache/chatterbox_onnx/
# Uses the smaller, quantized LLM by default.
synthesizer = ChatterboxOnnx(quantized=True) 
```

### 2\. Text-to-Speech (TTS)

Generate audio from text by cloning a voice provided via a reference WAV file. If `target_voice_path` is `None`, a default reference audio is downloaded and used.

| Parameter | Description |
| :--- | :--- |
| `text` | The input text to synthesize. |
| `target_voice_path` | Path to a WAV file of the target voice. **(Optional)** |
| `exaggeration` | Controls expressiveness (0.0 to 1.0). Default is 0.5. |
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

-----

### 3\. Voice Conversion (VC)

Convert the speech style and identity of a **source audio** file to match that of a **target voice reference**.

| Parameter | Description |
| :--- | :--- |
| `source_audio_path` | Path to the audio file containing the speech you want to convert. |
| `target_voice_path` | Path to the audio file of the voice identity you want to clone. |
| `output_file_name` | The path to save the converted WAV file. |

```python
synthesizer.voice_convert(
    source_audio_path="path/to/source_speech.wav",
    target_voice_path="path/to/target_voice_reference.wav",
    output_file_name="converted_voice.wav",
)
```

-----

### 4\. Batch Processing (TTS and VC)

#### Batch TTS Example

Generate the same text across all WAV files found in a specified folder, testing a range of `exaggeration` values.

```python
synthesizer.batch_synthesize(
    text="This is a test of the batch synthesis function.",
    voice_folder_path="path/to/folder_of_reference_voices",
    # (start, stop, step). Tests exaggeration values 0.3, 0.4, 0.5... 1.1.
    exaggeration_range=(0.3, 1.1, 0.1), 
    output_dir="batch_tts_results",
)
```

#### Batch VC Example

Convert a set of source audios using a set of reference voices.

```python
synthesizer.batch_voice_convert(
    original_audios_folder="path/to/source_audios", 
    voices_folder="path/to/reference_voices",  
    output_dir="batch_vc_results",
    n_random=2 # For each reference voice, convert 2 random source audios
)
```

-----

## 📚 Technical Details

The full set of models is sourced from the Hugging Face Hub: **[onnx-community/chatterbox-ONNX](https://huggingface.co/onnx-community/chatterbox-ONNX)**.

The pipeline comprises four key ONNX components:

1.  **`speech_encoder.onnx`**: Extracts speaker embeddings and speech tokens from a reference audio.
2.  **`embed_tokens.onnx`**: Converts text tokens into embeddings, applying the `exaggeration` feature.
3.  **`language_model[_q4].onnx`**: The core LLM (Llama-based) that performs auto-regressive generation of speech tokens, conditioned on text and speaker embeddings.
4.  **`conditional_decoder.onnx`**: The final neural vocoder that converts the sequence of generated speech tokens back into a high-fidelity waveform.
