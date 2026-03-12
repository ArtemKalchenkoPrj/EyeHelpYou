whisper_model = None
vision_model = None
router_model = None
command_model = None

def load_models():
    import whisper
    from langchain_ollama import ChatOllama
    global whisper_model
    global vision_model
    global router_model
    global command_model

    vision_model_name = "qwen3.5:397b-cloud"#"qwen3.5:397b-cloud""gemma3:27b-cloud"

    vision_model = ChatOllama(model=vision_model_name, temperature=0, format="json", reasoning=False)
    whisper_model = whisper.load_model("small")
    router_model = ChatOllama(model="ministral-3:14b-cloud", temperature=0, format="json", reasoning=False)
    command_model = ChatOllama(model="ministral-3:14b-cloud", temperature=0, format="json", reasoning=False)