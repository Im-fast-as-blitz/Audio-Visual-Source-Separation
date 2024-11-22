def normalize_audio(audio, loudness=20):
    """
    Normalize audio to desired loudness
    """
    return loudness * audio / audio.norm(dim=-1, keepdim=True)