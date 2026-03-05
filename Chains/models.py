whisper_model = None
mind_model = None

def load_models():
    import whisper
    from langchain_ollama import ChatOllama
    global whisper_model
    global mind_model

    mind_model = ChatOllama(model="qwen3.5:397b-cloud", temperature=0, format="json")
    whisper_model = whisper.load_model("small")