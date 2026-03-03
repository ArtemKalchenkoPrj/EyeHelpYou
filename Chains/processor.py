from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

import base64
def image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def run_llm(question: str, image_path: str) -> str:
    llm = ChatOllama(model="qwen3.5:397b-cloud", temperature=0)

    image_base64 = image_to_base64(image_path)

    system = """
    Ти - помічник для людей із вадами зору. Користувач надасть фото, якого фізично не можже побачити і питання, яке
    його цікавить. Твоя задача - відповісти на це питання максимально змістовно і коротко.
    - Не використовуй надто важких слів;
    - Розмовляй чемно;
    - Якщо питання стосується чогось небезпечного ти маєш застерегти користувача;
    - Відповідь може бути довжиною 250 символів максимум;
    """

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            },
            {
                "type": "text",
                "text": question
            }
        ])
    ]
    response = await llm.ainvoke(messages)
    print(response.content)
    return response.content