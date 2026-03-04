import edge_tts
import io

async def text_to_voice(text: str) -> io.BytesIO:
    communicate = edge_tts.Communicate(text, voice="uk-UA-OstapNeural")
    audio = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.write(chunk["data"])
    audio.seek(0)
    return audio