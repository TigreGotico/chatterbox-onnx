import os
import random

import librosa
import numpy as np
import onnxruntime
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Constants ---
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562


# --- Utility Class for Repetition Penalty ---

class RepetitionPenaltyLogitsProcessor:
    """
    Applies a repetition penalty to the logits.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Process logits based on input IDs and the penalty factor.
        """
        # Ensure input_ids is 2D (batch_size, sequence_length) for consistency
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]

        # Get the scores of the tokens that have already been generated
        score = np.take_along_axis(scores, input_ids, axis=1)

        # Apply penalty: if score < 0, multiply; otherwise, divide
        score = np.where(score < 0, score * self.penalty, score / self.penalty)

        # Update the scores with the penalized values
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


# --- Main Synthesizer Class ---

class ChatterboxOnnx:
    """
    A standalone class for performing text-to-speech synthesis using the
    Chatterbox ONNX models.
    """

    def __init__(self, quantized: bool = True,
                 cache_dir: str = os.path.expanduser("~/.cache/chatterbox_onnx")):
        """
        Initializes the synthesizer, downloads models, and creates ONNX sessions.

        Args:
            quantized: if True use Q4 quantized version of language model (350MB vs 2GB)
            cache_dir: Local directory to cache the downloaded ONNX files.
        """
        self.quantized = quantized
        self.model_id = "onnx-community/chatterbox-onnx"
        self.output_dir = cache_dir

        self.repetition_penalty = 1.2
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.repetition_penalty)

        print(f"Initializing ChatterboxSynthesizer. Model files will be cached in '{cache_dir}'...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, 'onnx'), exist_ok=True)

        self.tokenizer = self._load_tokenizer()

        # NOTE: Loading order is fixed here to match the assignment order below:
        # 1. speech_encoder, 2. embed_tokens, 3. language_model (llama), 4. conditional_decoder
        self.speech_encoder_session, \
            self.embed_tokens_session, \
            self.llama_with_past_session, \
            self.cond_decoder_session = self._load_models()

        self.num_hidden_layers = 30
        self.num_key_value_heads = 16
        self.head_dim = 64

    def _load_tokenizer(self):
        """Loads the AutoTokenizer from the Hugging Face model ID."""
        try:
            return AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

    def _download_and_get_session(self, filename: str) -> onnxruntime.InferenceSession:
        """Downloads an ONNX file and creates an InferenceSession."""
        path = hf_hub_download(
            repo_id=self.model_id,
            filename=filename,
            local_dir=self.output_dir,
            subfolder='onnx'
        )

        hf_hub_download(
            repo_id=self.model_id,
            filename=filename.replace(".onnx", ".onnx_data"),
            local_dir=self.output_dir,
            subfolder='onnx'
        )

        return onnxruntime.InferenceSession(path)

    def _load_models(self):
        """
        Downloads all ONNX model files and initializes ONNX runtime sessions.
        The order MUST match the assignment in __init__.
        """
        model_files = [
            "speech_encoder.onnx",  # -> speech_encoder_session
            "embed_tokens.onnx",  # -> embed_tokens_session
            "language_model_q4.onnx" if self.quantized else "language_model.onnx",
            "conditional_decoder.onnx"  # -> cond_decoder_session
        ]

        sessions = []
        for file in model_files:
            print(f"Loading {file}...")
            sessions.append(self._download_and_get_session(file))

        return sessions

    def _generate_waveform(
            self,
            text: str,
            audio_values: np.ndarray,
            max_new_tokens: int,
            exaggeration: float,
    ) -> np.ndarray:
        """
        Core generation loop that converts text and voice features into a waveform.
        Assumes audio_values are already loaded and prepped (np.float32, [1, N]).

        Returns:
            np.ndarray: The raw generated waveform (audio data).
        """
        # 1. Tokenize Text Input
        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)

        # Calculate position IDs for the text tokens
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1
        )

        ort_embed_tokens_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "exaggeration": np.array([exaggeration], dtype=np.float32)
        }

        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.long)

        # --- Generation Loop using kv_cache ---
        for i in tqdm(range(max_new_tokens), desc="Sampling Speech Tokens", dynamic_ncols=True):

            # --- Embed Tokens ---
            inputs_embeds = self.embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]

            if i == 0:
                # --- Run Speech Encoder for Context Embedding (Only on first step) ---
                ort_speech_encoder_input = {
                    "audio_values": audio_values,
                }
                speech_encoder_session = self.speech_encoder_session
                cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(None,
                                                                                               ort_speech_encoder_input)

                # Concatenate conditional embedding with text embeddings
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                # Prepare LLM inputs (Attention Mask and Past Key Values)
                batch_size, seq_len, _ = inputs_embeds.shape

                # Initialize Past Key Values (Empty cache)
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim],
                                                              dtype=np.float32)
                    for layer in range(self.num_hidden_layers)
                    for kv in ("key", "value")
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

            # --- Run Language Model (LLama) ---
            llama_with_past_session = self.llama_with_past_session
            logits, *present_key_values = llama_with_past_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **past_key_values,
            ))

            # Process Logits
            logits = logits[:, -1, :]  # Get logits for the last token
            next_token_logits = self.repetition_processor(generate_tokens[:, -1:], logits)

            # Sample next token (Greedy search: argmax)
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)

            # Check for stop token
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

            # Update inputs for next iteration
            position_ids = np.full((input_ids.shape[0], 1), i + 1, dtype=np.int64)
            ort_embed_tokens_inputs["input_ids"] = next_token
            ort_embed_tokens_inputs["position_ids"] = position_ids

            # Update Attention Mask and KV Cache
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

        print("Token generation complete.")

        # 2. Concatenate Speech Tokens and Run Conditional Decoder
        # Remove START and STOP tokens
        speech_tokens = generate_tokens[:, 1:-1]
        # Prepend prompt token
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

        cond_incoder_input = {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        }

        # Run the correct decoder session
        wav = self.cond_decoder_session.run(None, cond_incoder_input)[0]
        wav = np.squeeze(wav, axis=0)

        return wav

    def voice_convert(
            self,
            source_audio_path: str,
            target_voice_path: str,
            output_file_name: str = "converted_voice.wav",
    ):
        """
        Perform ONNX-based voice conversion using the Chatterbox ONNX models.
        This avoids using any PyTorch models and runs fully on ONNXRuntime.

        Args:
            source_audio_path: Path to the source voice audio.
            target_voice_path: Path to the target (reference) voice.
            output_file_name: Where to save the converted voice.
        """
        print("\n--- Starting ONNX Voice Conversion ---")
        print(f"Source: {source_audio_path}\nTarget: {target_voice_path}\nOutput: {output_file_name}")

        # Load source and target audio
        src_audio, _ = librosa.load(source_audio_path, sr=S3GEN_SR, res_type="soxr_hq")
        tgt_audio, _ = librosa.load(target_voice_path, sr=S3GEN_SR, res_type="soxr_hq")
        src_audio = src_audio[np.newaxis, :].astype(np.float32)
        tgt_audio = tgt_audio[np.newaxis, :].astype(np.float32)

        # --- Extract speaker embedding from target audio ---
        tgt_cond = {"audio_values": tgt_audio}
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.speech_encoder_session.run(None, tgt_cond)

        # --- Tokenize the source speech ---
        src_cond = {"audio_values": src_audio}
        _, src_tokens, _, _ = self.speech_encoder_session.run(None, src_cond)

        # Prepend target prompt token to source tokens for conditioning
        speech_tokens = np.concatenate([prompt_token, src_tokens], axis=1)

        cond_decoder_inputs = {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        }

        # --- Run conditional decoder ---
        print("Decoding waveform...")
        wav = self.cond_decoder_session.run(None, cond_decoder_inputs)[0]
        wav = np.squeeze(wav, axis=0)

        # Save result
        os.makedirs(os.path.dirname(output_file_name) or ".", exist_ok=True)
        sf.write(output_file_name, wav, S3GEN_SR)
        print(f"Voice conversion complete → {output_file_name}")

    def batch_voice_convert(
            self,
            original_audios_folder: str,
            voices_folder: str,
            output_dir: str = "batch_vc_output",
            n_random: int = 2,
    ):
        """
        Batch voice conversion: for each reference voice, select N random voices from folder_of_voices
        and perform cloning using ONNX pipeline.
        """
        print(f"\n--- Starting Batch Voice Conversion ---")
        os.makedirs(output_dir, exist_ok=True)

        # Gather reference and source voices
        src_files = [os.path.join(original_audios_folder, f) for f in os.listdir(original_audios_folder) if
                     f.lower().endswith('.wav')]
        ref_files = [os.path.join(voices_folder, f) for f in os.listdir(voices_folder) if
                     f.lower().endswith('.wav')]

        if not ref_files or not src_files:
            print("No valid .wav files found in input folders.")
            return

        for ref_path in tqdm(ref_files, desc="Reference voices"):
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
            selected_src = random.sample(src_files, min(n_random, len(src_files)))

            for src_path in selected_src:
                src_name = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{ref_name}_clone_{src_name}.wav"
                out_path = os.path.join(output_dir, out_name)

                try:
                    self.voice_convert(
                        source_audio_path=src_path,
                        target_voice_path=ref_path,
                        output_file_name=out_path,
                    )
                except Exception as e:
                    print(f"Error processing {src_name} -> {ref_name}: {e}")

    def synthesize(
            self,
            text: str,
            target_voice_path: str = None,
            max_new_tokens: int = 256,
            exaggeration: float = 0.5,
            output_file_name: str = "output.wav",
            apply_watermark: bool = False,
    ):
        """
        Runs the text-to-speech inference for a single voice and saves the audio.

        Args:
            text: The text to be synthesized.
            target_voice_path: Path to the reference audio file for the target voice.
                               If None, a default voice file is downloaded.
            max_new_tokens: The maximum number of speech tokens to generate.
            exaggeration: Controls the expressiveness of the generated speech (0.0 to 1.0).
            output_file_name: The path where the output WAV file will be saved.
            apply_watermark: If True, applies an audible watermark (requires resemble-perth).
        """
        print("\n--- Starting Text-to-Audio Inference ---")

        if not target_voice_path:
            target_voice_path = hf_hub_download(
                repo_id=self.model_id,
                filename="default_voice.wav",
                local_dir=self.output_dir
            )
            print(f"Using default voice: {target_voice_path}")

        # 1. Load Reference Audio
        try:
            # Use soxr_hq for high-quality resampling if needed
            audio_values, _ = librosa.load(target_voice_path, sr=S3GEN_SR, res_type='soxr_hq')
            audio_values = audio_values[np.newaxis, :].astype(np.float32)
        except Exception as e:
            print(f"Error loading target voice audio: {e}")
            return

        # 2. Generate Waveform
        wav = self._generate_waveform(text, audio_values, max_new_tokens, exaggeration)

        # 3. Optional: Apply Watermark
        if apply_watermark:
            print("Applying audio watermark...")
            try:
                import perth
                watermarker = perth.PerthImplicitWatermarker()
                wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)
            except ImportError:
                print("Warning: 'resemble-perth' not installed. Watermark skipped.")
            except Exception as e:
                print(f"Watermarking failed: {e}")

        # 4. Save Audio File
        sf.write(output_file_name, wav, S3GEN_SR)
        print(f"\nSuccessfully saved generated audio to: {output_file_name}")

    def batch_synthesize(
            self,
            text: str,
            voice_folder_path: str,
            exaggeration_range: tuple[float, float, float] = (0.5, 0.7, 0.1),  # (start, stop, step)
            max_new_tokens: int = 256,
            output_dir: str = "batch_output",
            apply_watermark: bool = False,
    ):
        """
        Performs batch text-to-speech synthesis using multiple reference voices
        and a range of exaggeration values.

        Args:
            text: The text to be synthesized.
            voice_folder_path: Path to the directory containing reference WAV files.
            exaggeration_range: A tuple (start, stop, step) defining the range of
                                exaggeration values to test. (e.g., (0.3, 0.9, 0.3) for 0.3, 0.6, 0.9)
                                If step is 0 or range is singular, only the start value is used.
            max_new_tokens: The maximum number of speech tokens to generate per file.
            output_dir: The directory where all generated WAV files will be saved.
            apply_watermark: If True, applies an audible watermark (requires resemble-perth).
        """
        print(f"\n--- Starting Batch Synthesis for text: '{text[:40]}...' ---")

        os.makedirs(output_dir, exist_ok=True)

        # 1. Prepare exaggeration values
        start, stop, step = exaggeration_range
        if step > 0 and start <= stop:
            exaggeration_values = np.arange(
                start,
                stop + step / 2,  # Add half step for float precision
                step
            ).round(2).tolist()
        else:
            exaggeration_values = [start]

        print(f"Testing exaggeration values: {exaggeration_values}")

        # 2. Find all WAV files
        try:
            voice_files = [
                os.path.join(voice_folder_path, f)
                for f in os.listdir(voice_folder_path)
                if f.lower().endswith('.wav')
            ]
        except FileNotFoundError:
            print(f"Error: Voice folder path not found: '{voice_folder_path}'. Aborting batch.")
            return

        if not voice_files:
            print(f"Error: No .wav files found in '{voice_folder_path}'. Aborting batch.")
            return

        total_generations = len(voice_files) * len(exaggeration_values)
        print(f"Found {len(voice_files)} reference voices. Will perform {total_generations} total generations.")

        # 3. Main Batch Loop
        for voice_path in voice_files:
            voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            print(f"\nProcessing voice: {voice_name}")

            # Load reference audio once per voice file
            try:
                # Use soxr_hq for high-quality resampling if needed
                audio_values, _ = librosa.load(voice_path, sr=S3GEN_SR, res_type='soxr_hq')
                audio_values = audio_values[np.newaxis, :].astype(np.float32)
            except Exception as e:
                print(f"  Skipping '{voice_path}' due to load error: {e}")
                continue

            for ex_val in exaggeration_values:
                print(f"  > Generating with exaggeration={ex_val:.2f}...")

                output_name = f"{voice_name}_exag{ex_val:.2f}.wav"
                output_file_path = os.path.join(output_dir, output_name)

                try:
                    # Generate Waveform
                    wav = self._generate_waveform(text, audio_values, max_new_tokens, ex_val)

                    # Optional: Apply Watermark
                    if apply_watermark:
                        try:
                            import perth
                            watermarker = perth.PerthImplicitWatermarker()
                            wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)
                        except ImportError:
                            print("  [Watermark] Warning: 'resemble-perth' not installed. Skipping.")
                        except Exception as e:
                            print(f"  [Watermark] Failed to apply: {e}")

                    # Save Audio File
                    sf.write(output_file_path, wav, S3GEN_SR)
                    print(f"  > Saved to {output_name}")

                except Exception as e:
                    print(f"  Error generating {output_name}: {e}")
                    continue

        print("\n--- Batch Synthesis Complete ---")

    def debug_info(self):
        """Print detailed ONNX session information and sample IO shapes."""

        def print_session_info(session, name):
            print(f"\n===== {name} =====")
            print("Providers:", session.get_providers())
            print("Inputs:")
            for inp in session.get_inputs():
                print(f" name={inp.name}, shape={inp.shape}, type={inp.type}")
            print("Outputs:")
            for out in session.get_outputs():
                print(f" name={out.name}, shape={out.shape}, type={out.type}")

        print("\n==================== DEBUG INFO ====================")
        print(f"Model ID: {self.model_id}")
        print(f"Cache directory: {self.output_dir}")

        sessions = [
            (self.speech_encoder_session, "Speech Encoder"),
            (self.embed_tokens_session, "Embed Tokens"),
            (self.llama_with_past_session, "Language Model"),
            (self.cond_decoder_session, "Conditional Decoder"),
        ]

        for sess, name in sessions:
            try:
                print_session_info(sess, name)
            except Exception as e:
                print(f"Error inspecting {name}: {e}")

        # Optional: Run a fake forward pass to inspect output shapes
        try:
            import librosa
            wav, _ = librosa.load(
                hf_hub_download(repo_id=self.model_id, filename="default_voice.wav", local_dir=self.output_dir),
                sr=24000)
            wav = wav[np.newaxis, :].astype(np.float32)
            print("\nRunning speech encoder on default voice for shape check...")
            outs = self.speech_encoder_session.run(None, {"audio_values": wav})
            for i, out in enumerate(outs):
                print(f" Output[{i}] shape={np.array(out).shape}, dtype={np.array(out).dtype}")
        except Exception as e:
            print(f"Could not run shape check: {e}")

        print("====================================================\n")



if __name__ == "__main__":
    # Note: The first run will download and cache all model files (approx. 5GB).

    AUDIOS = "/run/media/miro/endeavouros/synthww/hey_chatterbox"
    REFERENCE_VOICES = "/run/media/miro/endeavouros/dataset/not-wakeword/speech_samples"

    synthesizer = ChatterboxOnnx()  # Initialize the synthesizer (models are loaded here)
    synthesizer.debug_info()

    default_voice_path = f"{AUDIOS}/{os.listdir(AUDIOS)[0]}"
    target_voice_path= f"{REFERENCE_VOICES}/{os.listdir(REFERENCE_VOICES)[0]}"

    text = "The quick brown fox jumps over the lazy dog, demonstrating exceptional clarity and tone."

    # Example 1: TTS
    synthesizer.synthesize(
        text=text,
        # If target_voice_path is None, it uses a default reference audio.
        target_voice_path=None,
        exaggeration=0.7,
        output_file_name="chatterbox_output.wav",
        apply_watermark=False,
    )

    # Example 2: Voice clone
    synthesizer.voice_convert(
        source_audio_path=default_voice_path,
        target_voice_path=target_voice_path,
        output_file_name="converted_output.wav",
    )

    # Example 3: Voice clone folder of audios against folder of reference donor voices
    synthesizer.batch_voice_convert(
        original_audios_folder=AUDIOS, # convert from this
        voices_folder=REFERENCE_VOICES,  # to this
        output_dir="vc_results",
        n_random=2
    )

    # Example 4: TTS with multiple voices and exaggerations
    synthesizer.batch_synthesize(
        text=text,
        voice_folder_path=REFERENCE_VOICES,
        exaggeration_range=(0.3, 1.1, 0.1),
        output_dir="batch_results",
        apply_watermark=False,
    )
