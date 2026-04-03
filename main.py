"""
main.py — TurboQuant Local LLM Chat
=====================================
Run an LLM locally on your 8GB machine with TurboQuant KV cache compression.

Quick start:
    # With Ollama (easiest):
    ollama pull mistral
    python main.py

    # With HuggingFace:
    python main.py --backend huggingface --model microsoft/Phi-3-mini-4k-instruct

    # Change compression:
    python main.py --bits 3   # more compression, slightly lower quality
    python main.py --bits 4   # recommended (default)

    # Disable TurboQuant (baseline comparison):
    python main.py --no-compress

Commands during chat:
    /stats    — show compression statistics
    /clear    — clear conversation history
    /help     — show commands
    /quit     — exit
"""

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant running locally
via TurboQuant-compressed KV cache. You are concise, accurate, and honest
about uncertainty. For technical questions, you explain your reasoning."""


BANNER = """
╔══════════════════════════════════════════════════════╗
║           TurboQuant Local LLM                       ║
║      KV Cache Compression — ICLR 2026                ║
╠══════════════════════════════════════════════════════╣
║  Google Research • TurboQuantMSE + QJL Pipeline         ║
║  6x memory reduction • Zero accuracy loss            ║
╚══════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(description="TurboQuant Local LLM Chat")

    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Model backend (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Ollama: 'mistral', 'phi3', 'gemma2:2b'. "
            "HuggingFace: 'microsoft/Phi-3-mini-4k-instruct'"
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="TurboQuant bit-width (default: 4)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable TurboQuant (baseline mode)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Max context length in tokens (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    return parser.parse_args()


def setup_engine(args):
    """Load model and create inference engine based on CLI args."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import CONFIG
    from model.loader import load_huggingface_model, load_ollama_model
    from model.inference import HuggingFaceInference, OllamaInference

    # Apply CLI overrides to config
    CONFIG.backend = args.backend
    CONFIG.bits = args.bits
    CONFIG.enabled = not args.no_compress
    CONFIG.context_length = args.context
    CONFIG.temperature = args.temperature
    if args.hf_token:
        CONFIG.hf_token = args.hf_token

    if args.backend == "ollama":
        model_name = args.model or CONFIG.ollama_model
        logger.info(f"Loading Ollama model: {model_name}")
        client = load_ollama_model(model_name)
        return OllamaInference(client, config=CONFIG), "ollama"

    else:
        model_id = args.model or CONFIG.hf_model_id
        logger.info(f"Loading HuggingFace model: {model_id}")
        model, tokenizer = load_huggingface_model(
            model_id=model_id,
            bits=CONFIG.bits if CONFIG.enabled else 16,
            device=CONFIG.device,
            hf_token=CONFIG.hf_token,
            context_length=CONFIG.context_length,
        )
        return HuggingFaceInference(model, tokenizer, config=CONFIG), "huggingface"


def print_compression_info(args):
    """Print what TurboQuant is doing before chat starts."""
    if args.no_compress:
        print("  Mode: Standard (no compression)")
        return

    reduction = 16 // args.bits
    print(f"  TurboQuant: {args.bits}-bit KV cache (~{reduction}x memory reduction)")
    print(f"  Stage 1: TurboQuantMSE ({args.bits - 1}-bit, near-zero overhead)")
    print(f"  Stage 2: QJL (1-bit residual, unbiased estimator)")


def chat_loop(engine, backend: str):
    """Main interactive chat loop."""
    from model.inference import HuggingFaceInference, OllamaInference

    print("\nType your message. Commands: /stats /clear /help /quit\n")
    print("─" * 54)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break

            elif cmd == "/stats":
                if isinstance(engine, HuggingFaceInference):
                    print(engine.compression_report())
                else:
                    print("  Compression stats not available for Ollama backend.")
                continue

            elif cmd == "/clear":
                if isinstance(engine, OllamaInference):
                    engine.clear_history()
                elif isinstance(engine, HuggingFaceInference):
                    engine.kv_cache.clear()
                print("  Conversation cleared.")
                continue

            elif cmd == "/help":
                print("""
  Commands:
    /stats   — show TurboQuant compression statistics
    /clear   — clear conversation history and KV cache
    /quit    — exit
                """)
                continue

            else:
                print(f"  Unknown command: {user_input}")
                continue

        # ── Generate response ──────────────────────────────────────
        print("\nAssistant: ", end="", flush=True)

        try:
            if isinstance(engine, HuggingFaceInference):
                # Build prompt with system instruction
                prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_input} [/INST]"
                for token in engine.generate(prompt, stream=True):
                    print(token, end="", flush=True)

            elif isinstance(engine, OllamaInference):
                for token in engine.chat(
                    message=user_input,
                    system_prompt=SYSTEM_PROMPT,
                    stream=True,
                ):
                    print(token, end="", flush=True)

        except Exception as e:
            print(f"\n[Error during generation: {e}]")
            logger.exception("Generation error")

        print()   # newline after response


def main():
    args = parse_args()

    print(BANNER)
    print(f"  Backend:  {args.backend}")
    print(f"  Model:    {args.model or '(default)'}")
    print_compression_info(args)
    print()

    try:
        engine, backend = setup_engine(args)
    except ImportError as e:
        print(f"\n[Setup Error] Missing dependency: {e}")
        print("\nInstall dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Setup Error] {e}")
        logger.exception("Setup failed")
        sys.exit(1)

    print("\nModel ready. Starting chat...\n")
    chat_loop(engine, backend)


if __name__ == "__main__":
    main()