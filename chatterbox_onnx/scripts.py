import click
import os
from typing import Optional
from chatterbox_onnx import ChatterboxOnnx

# Global variable to hold the synthesizer instance.
# It's initialized once on the first command run.
synthesizer: Optional[ChatterboxOnnx] = None


def get_synthesizer():
    """Initializes and returns the global ChatterboxOnnx instance."""
    global synthesizer
    if synthesizer is None:
        # Note: The first run will download and cache all model files (approx. 5GB).
        click.echo("Initializing ChatterboxOnnx synthesizer... (This may take a moment)")
        synthesizer = ChatterboxOnnx()
        synthesizer.debug_info()
    return synthesizer


@click.command(name="bulk-vc")
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
    help="Path to the directory where converted audio files will be saved.",
)
@click.option(
    "-a",
    "--audios-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, resolve_path=True),
    help="Path to the folder containing the source audio files to convert.",
)
@click.option(
    "-v",
    "--voices-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, resolve_path=True),
    help="Path to the folder containing the reference donor voice files (.wav).",
)
@click.option(
    "-n",
    "--n-random",
    default=2,
    type=int,
    help="Number of random voices to use for cloning from the voices-path (per source audio file).",
    show_default=True,
)
def bulk_vc(output_path: str, audios_path: str, voices_path: str, n_random: int):
    """Voice clones a folder of audios against a folder of reference voices."""

    synthesizer = get_synthesizer()

    click.echo(f"Starting Bulk Voice Conversion (VC)...")
    click.echo(f"  Source Audios: {audios_path}")
    click.echo(f"  Reference Voices: {voices_path}")
    click.echo(f"  Output Directory: {output_path}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Voice clone folder of audios against folder of reference donor voices
    synthesizer.batch_voice_convert(
        original_audios_folder=audios_path,
        voices_folder=voices_path,
        output_dir=output_path,
        n_random=n_random
    )

    click.echo("\nBulk Voice Conversion (VC) Complete! ✨")


@click.command(name="bulk-tts")
@click.argument("text", type=str)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
    help="Path to the directory where synthesized audio files will be saved.",
)
@click.option(
    "-v",
    "--voices-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, resolve_path=True),
    help="Path to the folder containing the reference voice files for TTS.",
)
@click.option(
    "-e",
    "--exaggeration-range",
    nargs=3,  # Requires exactly 3 arguments (start, stop, step)
    type=float,
    default=(0.3, 1.0, 0.1),
    help="Range (start stop step) for voice style exaggeration. Example: '0.3 1.0 0.1'.",
    show_default=True,
)
def bulk_tts(text: str, output_path: str, voices_path: str, exaggeration_range: tuple[float, float, float]):
    """Performs Text-to-Speech (TTS) using a folder of reference voices and a text string."""

    synthesizer = get_synthesizer()

    click.echo(f"Starting Bulk Text-to-Speech (TTS)...")
    click.echo(f"  Text: '{text}'")
    click.echo(f"  Reference Voices: {voices_path}")
    click.echo(f"  Exaggeration Range: {exaggeration_range}")
    click.echo(f"  Output Directory: {output_path}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # TTS with multiple voices and exaggerations
    synthesizer.batch_synthesize(
        text=text,
        voice_folder_path=voices_path,
        exaggeration_range=exaggeration_range,
        output_dir=output_path,
        apply_watermark=False,
    )

    click.echo("\nBulk Text-to-Speech (TTS) Complete! 🎤")

