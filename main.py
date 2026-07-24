import asyncio
import base64
import contextlib
import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# -------------------------------------------------------------------
# 1. Logging & Environment Setup
# -------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# httpx logs every single outgoing HTTP call at INFO level, which drowns
# out the bot's own logs. Quiet it down to warnings and above.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not TELEGRAM_TOKEN or not OPENROUTER_KEY:
    raise ValueError(
        "❌ Missing API Keys! Please check your .env file for TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY."
    )

# FIX: the original used the *sync* `OpenAI` client and called
# `.chat.completions.create(...)` with no `await` inside an `async def`
# handler. A sync network call like that blocks asyncio's single event
# loop for its entire duration — every other user, and every other
# message, would freeze until that one API call returned. AsyncOpenAI +
# `await` (used below) fixes this so the bot can serve many users at once.
ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    timeout=90.0,
    max_retries=2,
)

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://telegram-bot.local",
    "X-Title": "Telegram AI Assistant",
}

# -------------------------------------------------------------------
# 2. Configuration & State Management
# -------------------------------------------------------------------
AVAILABLE_MODELS = {
    "⚡ Gemini 2.0 Flash (Fast & Free)": "google/gemini-2.0-flash-001",
    "🧠 GPT-4o Mini (Smart)": "openai/gpt-4o-mini",
    "🚀 Claude 3.5 Sonnet (Advanced)": "anthropic/claude-3.5-sonnet",
    "🦙 LLaMA 3.3 70B (Open Source)": "meta-llama/llama-3.3-70b-instruct",
}
MODEL_LABELS = {model_id: label for label, model_id in AVAILABLE_MODELS.items()}

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
MAX_HISTORY = 10  # non-system messages kept per user (5 user+assistant pairs)
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TELEGRAM_MAX_LENGTH = 4096
DEFAULT_PHOTO_PROMPT = (
    "Is image ko ache se analyze karein, isme diye sawal ko solve karein "
    "aur step-by-step explain karein."
)

user_models = defaultdict(lambda: DEFAULT_MODEL)
user_chat_history = defaultdict(list)
# FIX: one lock per user serializes that user's own requests, so two
# messages sent back-to-back can't interleave and corrupt shared history,
# and replies always come back in the order the messages were sent.
user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an expert AI assistant and tutor integrated into a Telegram bot. "
        "Provide clear, accurate, and structured answers in Hindi, Hinglish, or English based on user query. "
        "When solving math, science problems, or reading text from photos, provide step-by-step solutions."
    ),
}


# -------------------------------------------------------------------
# 3. Helpers
# -------------------------------------------------------------------
def trim_history(history: list) -> list:
    """Keep the system prompt plus the last MAX_HISTORY messages.

    FIX: the original trimmed with `[SYSTEM_PROMPT] + history[-(MAX_HISTORY):]`,
    a plain slice that can cut a (user, assistant) pair in half and leave
    the system prompt followed directly by an *assistant* turn. Several
    models on OpenRouter (Claude in particular) reject that shape outright
    because they require strict user/assistant alternation. This version
    always trims in whole pairs and re-anchors on a user turn.
    """
    if len(history) <= MAX_HISTORY + 1:
        return history
    convo = history[1:][-MAX_HISTORY:]
    if convo and convo[0]["role"] != "user":
        convo = convo[1:]
    return [history[0]] + convo


def split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list:
    """Split text into Telegram-safe chunks.

    FIX: Telegram rejects any message over 4096 characters outright, and
    the original sent the AI's raw answer in a single `reply_text` call
    with no length check — any detailed step-by-step answer over that
    limit would crash with a "Message is too long" error and the user
    would get nothing at all.
    """
    if len(text) <= limit:
        return [text]
    chunks = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1 or split_at < limit * 0.5:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    if text:
        chunks.append(text)
    return chunks


async def send_answer(message, text: str) -> None:
    """Send a (possibly long) AI answer, replying with the first chunk.

    FIX: the original always sent with no parse_mode, so any markdown the
    AI produced (**bold**, lists, etc.) showed up to the user as raw
    asterisks. This tries Markdown first and falls back to plain text
    per-chunk if the AI's output has unbalanced entities Telegram can't
    parse (a common real-world failure with free-form model output).
    """
    for i, chunk in enumerate(split_message(text)):
        sender = message.reply_text if i == 0 else message.chat.send_message
        try:
            await sender(chunk, parse_mode="Markdown")
        except Exception:
            await sender(chunk)


async def _typing_loop(chat) -> None:
    """Keep the 'typing…' indicator alive — Telegram clears it after ~5s,
    but a slow model call can easily take longer than that."""
    try:
        while True:
            await chat.send_action(action="typing")
            await asyncio.sleep(4)
    except (asyncio.CancelledError, Exception):
        pass


async def with_typing(chat, coro):
    """Run `coro` while showing a continuously-refreshed typing indicator."""
    task = asyncio.create_task(_typing_loop(chat))
    try:
        return await coro
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def sanitize_error(e: Exception) -> str:
    text = str(e) or e.__class__.__name__
    return text if len(text) <= 250 else text[:250] + "…"


# -------------------------------------------------------------------
# 4. Command Handlers
# -------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_text = (
        "👋 **Namaste! Main aapka AI Assistant hoon.**\n\n"
        "Main aapke in sabhi kaam mein help kar sakta hoon:\n"
        "• 💬 **Sawal-Jawab & Chat**: Kuch bhi puchiye.\n"
        "• 📷 **Photo Solver**: Homework, Math question, ya Code ki pic bhejiye solution paane ke liye.\n"
        "• 🔀 **Model Switch**: `/model` command se AI model badlein.\n"
        "• 🧹 **Reset Chat**: `/reset` command se purani chat clear karein.\n\n"
        "Shuru karne ke liye koi message ya photo bhejiye!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "💡 **Kaise Use Karein:**\n\n"
        "1. **Text Message**: Apna sawal normal type karke bhejiye.\n"
        "2. **Photo**: Sawaal ki photo bhejiye. Caption mein instruction bhi likh sakte hain.\n"
        "3. **Change Model**: `/model` type karke doosra AI model select karein.\n"
        "4. **Clear History**: Chat reset karne ke liye `/reset` type karein."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_chat_history[user_id].clear()
    await update.message.reply_text(
        "🧹 **Conversation history clear ho gayi hai!**", parse_mode="Markdown"
    )


async def model_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    current_model = user_models[user_id]
    current_label = MODEL_LABELS.get(current_model, current_model)

    keyboard = [
        [
            InlineKeyboardButton(
                f"✅ {label}" if model_id == current_model else label,
                callback_data=f"set_model:{model_id}",
            )
        ]
        for label, model_id in AVAILABLE_MODELS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"⚙️ **Current Active Model**: {current_label}\n\nNeeche se naya model chuniye:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


async def model_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    _, selected_model = query.data.split(":", 1)
    user_id = query.from_user.id
    user_models[user_id] = selected_model
    label = MODEL_LABELS.get(selected_model, selected_model)

    await query.edit_message_text(
        f"✅ **Active model set to:**\n{label}",
        parse_mode="Markdown",
    )


# -------------------------------------------------------------------
# 5. Message Handlers (Text & Image)
# -------------------------------------------------------------------
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_prompt = update.message.text
    active_model = user_models[user_id]

    async with user_locks[user_id]:
        history = user_chat_history[user_id]
        if not history:
            history.append(SYSTEM_PROMPT)
        history.append({"role": "user", "content": user_prompt})

        try:
            response = await with_typing(
                update.message.chat,
                ai_client.chat.completions.create(
                    model=active_model,
                    messages=history,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    extra_headers=OPENROUTER_HEADERS,
                ),
            )
            if not response.choices:
                raise ValueError("Model returned no choices")
            answer = (
                response.choices[0].message.content
                or "⚠️ Model se khaali jawab mila, dobara try karein."
            )
        except Exception as e:
            history.pop()  # undo the user turn we never got an answer for
            logger.exception("Text error for user %s", user_id)
            await update.message.reply_text(
                f"❌ Kuch gadbad ho gayi: {sanitize_error(e)}\n"
                "Dobara try karein, ya /model se doosra model select karein."
            )
            return

        history.append({"role": "assistant", "content": answer})
        user_chat_history[user_id] = trim_history(history)

        try:
            await send_answer(update.message, answer)
        except Exception:
            logger.exception("Delivery error for user %s", user_id)


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    active_model = user_models[user_id]
    caption_text = update.message.caption or DEFAULT_PHOTO_PROMPT

    async with user_locks[user_id]:
        history = user_chat_history[user_id]
        if not history:
            history.append(SYSTEM_PROMPT)

        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(photo_bytes).decode("utf-8")

            # Build a separate list for the API call instead of appending the
            # base64 image straight into `history` — this keeps raw image
            # bytes out of stored history (avoids bloating memory per user
            # across many photos), while the lightweight text record added
            # below still keeps follow-up questions in context.
            vision_messages = history + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            response = await with_typing(
                update.message.chat,
                ai_client.chat.completions.create(
                    model=active_model,
                    messages=vision_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    extra_headers=OPENROUTER_HEADERS,
                ),
            )
            if not response.choices:
                raise ValueError("Model returned no choices")
            answer = (
                response.choices[0].message.content
                or "⚠️ Model se khaali jawab mila, dobara try karein."
            )
        except Exception as e:
            logger.exception("Photo error for user %s", user_id)
            await update.message.reply_text(f"❌ Image process nahi ho payi: {sanitize_error(e)}")
            return

        # FIX: the original never recorded photo Q&A in history, so a
        # text follow-up like "isme step 2 samjhao" had zero context about
        # the photo that was just solved. Store a short text record of it.
        history.append({"role": "user", "content": f"[📷 Photo] {caption_text}"})
        history.append({"role": "assistant", "content": answer})
        user_chat_history[user_id] = trim_history(history)

        try:
            await send_answer(update.message, answer)
        except Exception:
            logger.exception("Delivery error for user %s", user_id)


# -------------------------------------------------------------------
# 6. Global Error Handler
# -------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catches anything that slips past the per-handler try/excepts (e.g.
    errors raised by python-telegram-bot itself) so the bot logs instead
    of silently dropping the update."""
    logger.error("Unhandled exception while processing an update", exc_info=context.error)


# -------------------------------------------------------------------
# 7. Main Execution
# -------------------------------------------------------------------
def main() -> None:
    print("🚀 Initializing Telegram Bot...")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .concurrent_updates(True)  # FIX: lets multiple users be served at once
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("model", model_menu_command))

    # Callbacks & Messages
    app.add_handler(CallbackQueryHandler(model_callback_handler, pattern="^set_model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_error_handler(error_handler)

    print("🤖 Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

