import asyncio
import logging
import os
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

dp = Dispatcher()
bot: Bot | None = Bot(token=TOKEN) if TOKEN else None


def clean_answer(text: str) -> str:
    return re.sub(r"\[src:[^\]]+\]", "", text).strip()


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Давай же начнем наше общение! Я всегда на связи, спрашивай 💙",
        parse_mode=None,
    )


@dp.message(F.text)
async def on_message(message: Message) -> None:
    question = message.text or ""
    log.info("user=%s q=%r", message.from_user.id if message.from_user else "?", question[:80])

    await bot.send_chat_action(message.chat.id, "typing")

    def _answer():
        # Import inside thread — never blocks the event loop
        from scripts.rag import ask_question
        return ask_question(question)

    answer = await asyncio.to_thread(_answer)
    await message.answer(clean_answer(answer), parse_mode=None)


async def main() -> None:
    if not bot:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    log.info("kAI bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
