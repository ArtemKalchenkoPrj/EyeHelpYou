import os
from aiogram import Bot, Dispatcher
from Telegram.handlers import user

async def main():
    from dotenv import load_dotenv
    load_dotenv()

    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    dp = Dispatcher()
    dp.include_router(user)
    await dp.start_polling(bot)

if __name__ == "__main__":
    from Chains.processor import run_llm
    import asyncio
    asyncio.run(run_llm("Що зображено на фото?","C:/Users/2004a/Downloads/images.webp"))


"""if __name__ == '__main__':
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        pass"""
