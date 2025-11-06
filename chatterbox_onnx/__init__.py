import os
import random

import glob
import librosa
import numpy as np
import onnxruntime
import soundfile as sf
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

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
        Initialize the ChatterboxOnnx synthesizer and prepare tokenizer, model files, and ONNX inference sessions.

        Parameters:
            quantized (bool): If True, use the smaller Q4 quantized language model binary; otherwise use the full model.
            cache_dir (str): Local directory where model files and tokenizer are cached and where ONNX files are stored.
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

        # These parameters should match the LLM's architecture
        self.num_hidden_layers = 30
        self.num_key_value_heads = 16
        self.head_dim = 64

    def _load_tokenizer(self) -> Tokenizer:
        """
        Load the model's tokenizer.json from the Hugging Face Hub and return a Tokenizer instance.

        Downloads the tokenizer.json for the configured model into the instance's output directory and constructs a Tokenizer from that file.

        Returns:
            Tokenizer: The Tokenizer loaded from the downloaded tokenizer.json.
        """
        try:
            # 1. Download the tokenizer.json file
            tokenizer_path = hf_hub_download(
                repo_id=self.model_id,
                filename="tokenizer.json",
                local_dir=self.output_dir
            )

            # 2. Load the tokenizer using the dedicated tokenizers library
            return Tokenizer.from_file(tokenizer_path)

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
        Download the Chatterbox ONNX model files in the expected order and create ONNX Runtime InferenceSession objects.

        Returns:
            sessions (list): List of onnxruntime.InferenceSession objects in the same order as model_files:
                [speech_encoder_session, embed_tokens_session, language_model_session (quantized if enabled), cond_decoder_session].
        Note:
            The ordering of returned sessions matches the assignments performed in __init__ and must be preserved.
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

    def _generate_waveform(self, text: str,
                           cond_emb, prompt_token, ref_x_vector, prompt_feat,
                           max_new_tokens: int,
                           exaggeration: float,
                           speech_tokens=None):

        """
         Generate a waveform conditioned on text and speaker embeddings, optionally generating missing speech tokens.

         Parameters:
            text (str): Input text prompt used for token generation when `speech_tokens` is not provided.
            cond_emb (np.ndarray): Conditional embedding tensor produced by the speech encoder to prepend to token embeddings.
            prompt_token (np.ndarray): Token sequence to prepend to generated speech tokens (already shaped for batching).
            ref_x_vector (np.ndarray): Speaker embedding(s) used by the conditional decoder.
            prompt_feat (np.ndarray): Speaker feature tensor used by the conditional decoder.
            max_new_tokens (int): Maximum number of speech tokens to generate when `speech_tokens` is not provided.
            exaggeration (float): Scalar controlling prosody/exaggeration applied to token embeddings.
            speech_tokens (optional, np.ndarray): Precomputed speech token sequence to bypass token generation; if None, tokens are generated from `text`.

         Returns:
            np.ndarray: 1-D waveform array (float32) sampled at 24000 Hz.
        """
        if speech_tokens is None:
            # 1. Tokenize Text Input
            encoding = self.tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)

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

            generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

            # --- Generation Loop using kv_cache ---
            for i in range(max_new_tokens):

                # --- Embed Tokens ---
                inputs_embeds = self.embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]

                if i == 0:
                    # Concatenate conditional embedding with text embeddings
                    inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                    # Prepare LLM inputs (Attention Mask and Past Key Values)
                    batch_size, seq_len, _ = inputs_embeds.shape

                    # Initialize Past Key Values (Empty cache)
                    past_key_values = {
                        f"past_key_values.{layer}.{kv}": np.zeros(
                            [batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
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

    def embed_speaker(self, source_audio_path: str):
        # --- Extract speaker embedding from audio ---
        """
        Extract speaker conditioning embeddings, prompt token, speaker embedding vector, and speaker features from a local audio file.

        Parameters:
            source_audio_path (str): Path to the source audio file to embed.

        Returns:
            cond_emb (np.ndarray): Conditioning embedding produced by the speech encoder.
            prompt_token: Prompt token(s) returned by the speech encoder for conditioning/generation.
            ref_x_vector (np.ndarray): Speaker embedding vector representing speaker characteristics.
            prompt_feat (np.ndarray): Additional speaker feature descriptors used by the conditional decoder.
        """
        src_audio, _ = librosa.load(source_audio_path, sr=S3GEN_SR, res_type="soxr_hq")
        src_audio = src_audio[np.newaxis, :].astype(np.float32)
        tgt_cond = {"audio_values": src_audio}
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.speech_encoder_session.run(None, tgt_cond)
        return cond_emb, prompt_token, ref_x_vector, prompt_feat

    def _watermark_and_save(self, wav, output_file_name: str, apply_watermark=False):

        # 3. Optional: Apply Watermark
        """
        Optionally apply an implicit watermark to an audio waveform and save it to disk.

        Parameters:
            wav (np.ndarray): 1-D audio samples at the module sample rate (S3GEN_SR).
            output_file_name (str): Destination file path for the saved WAV.
            apply_watermark (bool): If True, attempt to apply an implicit watermark using the `perth` library;
                if `perth` is not available or watermarking fails, the original audio is saved unmodified.

        Returns:
            str: The path to the saved output file (same as `output_file_name`).
        """
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
        return output_file_name

    def voice_convert(
            self,
            source_audio_path: str,
            target_voice_path: str,
            output_file_name: str = "converted_voice.wav",
            exaggeration: float = 0.6,
            max_new_tokens: int = 512,
            apply_watermark=False
    ):
        """
        Convert a source audio file to sound like a target (reference) voice and save the converted audio.

        Parameters:
            source_audio_path (str): Path to the source audio file to be converted.
            target_voice_path (str): Path to the reference audio file whose voice characteristics will be applied.
            output_file_name (str): Filename (or path) where the converted audio will be written.
            exaggeration (float): Factor controlling how strongly the target voice characteristics are applied.
            max_new_tokens (int): Maximum number of speech tokens to generate during waveform synthesis.
            apply_watermark (bool): If True, apply an audio watermark to the output when possible.

        Returns:
            output_path (str): Path to the saved converted audio file.
        """
        print("\n--- Starting ONNX Voice Conversion ---")
        print(f"Source: {source_audio_path}\nTarget: {target_voice_path}\nOutput: {output_file_name}")

        # --- Extract speaker embedding from target audio ---
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(target_voice_path)

        # --- Tokenize the source speech ---
        _, src_tokens, _, _ = self.embed_speaker(source_audio_path)

        # Prepend target prompt token to source tokens for conditioning
        speech_tokens = np.concatenate([prompt_token, src_tokens], axis=1)

        wav = self._generate_waveform(text="",
                                      cond_emb=cond_emb,
                                      prompt_token=src_tokens,
                                      ref_x_vector=ref_x_vector,
                                      prompt_feat=prompt_feat,
                                      max_new_tokens=max_new_tokens,
                                      exaggeration=exaggeration,
                                      speech_tokens=speech_tokens)
        return self._watermark_and_save(wav, output_file_name, apply_watermark)

    def batch_voice_convert(
            self,
            original_audios_folder: str,
            voices_folder: str,
            output_dir: str = "batch_vc_output",
            n_random: int = 2,
    ):
        """
        Perform batch voice cloning by converting selected source WAV files to each reference voice.

        Parameters:
            original_audios_folder (str): Path to a folder containing source .wav files to be converted.
            voices_folder (str): Path to a folder containing reference .wav files that provide target voices.
            output_dir (str): Directory where converted files will be written; created if it does not exist.
            n_random (int): Number of random source files to convert for each reference voice (capped at the number of available sources).
        """
        print(f"\n--- Starting Batch Voice Conversion ---")
        os.makedirs(output_dir, exist_ok=True)

        # Gather reference and source voices
        src_files = glob.glob(os.path.join(original_audios_folder, '**', '*.wav'), recursive=True)
        ref_files = glob.glob(os.path.join(voices_folder, '**', '*.wav'), recursive=True)

        if not ref_files or not src_files:
            print("No valid .wav files found in input folders.")
            return

        for ref_path in ref_files:
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
            target_voice_path: str = None,  # TODO - allow passing embeddings directly alternatively
            max_new_tokens: int = 512,
            exaggeration: float = 0.5,
            output_file_name: str = "output.wav",
            apply_watermark: bool = False,
    ):
        """
        Synthesize speech from text using a target voice and save the resulting WAV file.

        Parameters:
            text (str): Text to synthesize.
            target_voice_path (str | None): Path to a reference audio file for the target voice. If None, a default voice file is downloaded and used.
            max_new_tokens (int): Maximum number of speech tokens to generate.
            exaggeration (float): Controls expressiveness of the generated speech; values typically range from 0.0 (neutral) to 1.0 (highly expressive).
            output_file_name (str): Path where the output WAV file will be written.
            apply_watermark (bool): If True, attempt to apply an audible watermark before saving; if watermarking is unavailable, the file is saved without it.
        """
        print("\n--- Starting Text-to-Audio Inference ---")

        if not target_voice_path:
            target_voice_path = hf_hub_download(
                repo_id=self.model_id,
                filename="default_voice.wav",
                local_dir=self.output_dir
            )
            print(f"Using default voice: {target_voice_path}")

        # 2. Generate Waveform
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(target_voice_path)
        wav = self._generate_waveform(text,
                                      cond_emb, prompt_token, ref_x_vector, prompt_feat,
                                      max_new_tokens, exaggeration)
        self._watermark_and_save(wav, output_file_name, apply_watermark)

    def batch_synthesize(
            self,
            text: str,
            voice_folder_path: str,
            exaggeration_range: tuple[float, float, float] = (0.5, 0.7, 0.1),  # (start, stop, step)
            max_new_tokens: int = 512,
            output_dir: str = "batch_output",
            apply_watermark: bool = False,
    ):
        """
        Perform batch text-to-speech synthesis using multiple reference voices and a range of exaggeration values.

        Parameters:
            text (str): The text to synthesize for every reference voice and exaggeration setting.
            voice_folder_path (str): Path to a directory containing reference `.wav` files to use as target voices.
            exaggeration_range (tuple[float, float, float]): `(start, stop, step)` defining exaggeration values to test.
                If `step > 0` and `start <= stop`, values from `start` to `stop` (inclusive, stepped by `step`) are used;
                otherwise only `start` is used.
            max_new_tokens (int): Maximum number of speech tokens to generate per synthesis pass.
            output_dir (str): Directory where generated WAV files will be written; created if it does not exist.
            apply_watermark (bool): If True, attempts to apply an audible watermark using the `perth` package before saving.
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
        voice_files = glob.glob(os.path.join(voice_folder_path, '**', '*.wav'), recursive=True)

        if not voice_files:
            print(f"Error: No .wav files found in '{voice_folder_path}'. Aborting batch.")
            return

        total_generations = len(voice_files) * len(exaggeration_values)
        print(f"Found {len(voice_files)} reference voices. Will perform {total_generations} total generations.")

        # 3. Main Batch Loop
        for voice_path in voice_files:
            voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            print(f"\nProcessing voice: {voice_name}")

            cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(voice_path)

            for ex_val in exaggeration_values:
                print(f"  > Generating with exaggeration={ex_val:.2f}...")

                output_name = f"{voice_name}_exag{ex_val:.2f}.wav"
                output_file_path = os.path.join(output_dir, output_name)

                try:
                    # Generate Waveform
                    wav = self._generate_waveform(text,
                                                  cond_emb, prompt_token, ref_x_vector, prompt_feat,
                                                  max_new_tokens, ex_val)
                    self._watermark_and_save(wav, output_file_path, apply_watermark)

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
