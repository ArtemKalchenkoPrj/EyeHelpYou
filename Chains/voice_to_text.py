from groq import AsyncGroq

async def voice_to_text(file_path: str) -> str:
    client = AsyncGroq()
    with open(file_path, "rb") as audio_file:
        transcription = await client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
        )
    return transcription.text

