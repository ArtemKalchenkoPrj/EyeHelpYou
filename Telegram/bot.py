from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from commands import start, help_command
from messages import echo
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Реєстрація хендлерів
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Бот запущено...")
    app.run_polling()

if __name__ == "__main__":
    main()