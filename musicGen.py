import torchaudio
from audiocraft.models import MusicGen
from audiocraft.models import AudioGen
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write
import torch
        

def gen_audio():
    # https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    model.set_generation_params(duration=5)  # generate 5 seconds.
    descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
    wav = model.generate(descriptions)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}_audio', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

def gen_diffusion_and_melodies():
    # https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    mbd = MultiBandDiffusion.get_mbd_musicgen()
    model.set_generation_params(duration=8)  # generate 8 seconds.
    wav, tokens = model.generate_unconditional(4, return_tokens=True)    # generates 4 unconditional audio samples and keep the tokens for MBD generation
    descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
    wav_diffusion = mbd.tokens_to_wav(tokens)
    wav, tokens = model.generate(descriptions, return_tokens=True)  # generates 3 samples and keep the tokens.
    wav_diffusion = mbd.tokens_to_wav(tokens)
    melody, sr = torchaudio.load('./assets/bach.mp3')
    # Generates using the melody from the given audio and the provided descriptions, returns audio and audio tokens.
    wav, tokens = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr, return_tokens=True)
    wav_diffusion = mbd.tokens_to_wav(tokens)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav and {idx}_diffusion.wav, with loudness normalization at -14 db LUFS for comparing the methods.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        audio_write(f'{idx}_diffusion', wav_diffusion[idx].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

def gen_Magnet():
    # This model requires a GPU. 
    # https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md
    model = MAGNeT.get_pretrained('facebook/magnet-small-10secs')
    descriptions = ['disco beat', 'energetic EDM', 'funky groove']
    wav = model.generate(descriptions)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}_magnet', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

gen_diffusion_and_melodies()
gen_audio()
if torch.cuda.is_available():
    gen_Magnet()

