from groq import AsyncGroq
import io

async def voice_to_text(audio_buffer: io.BytesIO) -> str:
    client = AsyncGroq()
    transcription = await client.audio.transcriptions.create(
        file=("voice.ogg", audio_buffer.read()),
        model="whisper-large-v3-turbo",
        temperature=0,
        response_format="verbose_json",
    )
    return transcription.text