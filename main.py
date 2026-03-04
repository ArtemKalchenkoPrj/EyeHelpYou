import os
from aiogram import Bot, Dispatcher
from Telegram.handlers import user

async def main():
    from dotenv import load_dotenv
    load_dotenv()

    bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
    dp = Dispatcher()
    dp.include_router(user)
    print("Bot started")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
