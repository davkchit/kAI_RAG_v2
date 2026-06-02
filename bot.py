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

    try:
        def _answer():
            from scripts.rag import ask_question
            return ask_question(question)

        answer = await asyncio.wait_for(asyncio.to_thread(_answer), timeout=60)
        await message.answer(clean_answer(answer), parse_mode=None)

    except asyncio.TimeoutError:
        log.error("Timeout after 60s for q=%r", question[:80])
        await message.answer("Не успел обработать запрос, попробуй ещё раз.", parse_mode=None)
    except Exception as e:
        log.error("Error answering q=%r: %s", question[:80], e, exc_info=True)
        await message.answer("Произошла ошибка, попробуй ещё раз.", parse_mode=None)


async def main() -> None:
    if not bot:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    log.info("kAI bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
