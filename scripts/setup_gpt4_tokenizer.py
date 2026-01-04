"""
Setup pretrained tokenizer for nanochat.

This script creates the necessary tokenizer files so you can skip
training your own tokenizer and use a pretrained one instead.

It adds nanochat's special tokens to the base tokenizer:
    - <|bos|>, <|user_start|>, <|user_end|>, etc.

Usage:
    python -m scripts.setup_gpt4_tokenizer          # 默认使用 GPT-2
    python -m scripts.setup_gpt4_tokenizer --gpt2   # 使用 GPT-2 (50,257 tokens)
    python -m scripts.setup_gpt4_tokenizer --gpt4   # 使用 GPT-4 (100,277 tokens)

This will create:
    - $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
    - $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt
"""

import argparse
import os
import pickle

import tiktoken
import torch

from nanochat.common import get_base_dir
from nanochat.tokenizer import SPECIAL_TOKENS


def main():
    parser = argparse.ArgumentParser(description="Setup pretrained tokenizer for nanochat")
    parser.add_argument("--gpt2", action="store_true", help="Use GPT-2 tokenizer (50,257 tokens)")
    parser.add_argument("--gpt4", action="store_true", help="Use GPT-4 tokenizer (100,277 tokens)")
    args = parser.parse_args()

    # 默认使用 GPT-2（更小的词表）
    if args.gpt4:
        encoding_name = "cl100k_base"
        friendly_name = "GPT-4"
    else:
        encoding_name = "gpt2"
        friendly_name = "GPT-2"

    # 配置路径
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"Tokenizer directory: {tokenizer_dir}")

    # 1. 加载 base tokenizer
    print(f"Loading {friendly_name} tokenizer ({encoding_name})...")
    base_enc = tiktoken.get_encoding(encoding_name)
    print(f"Base vocab size: {base_enc.n_vocab}")

    # 2. 添加 nanochat 的特殊 token
    print(f"Adding nanochat special tokens: {SPECIAL_TOKENS}")
    # 获取 tokenizer 的 mergeable_ranks 和现有 special_tokens
    mergeable_ranks = base_enc._mergeable_ranks
    # 在现有词表末尾添加 nanochat 的特殊 token
    tokens_offset = base_enc.n_vocab
    special_tokens = dict(base_enc._special_tokens)  # 复制现有的
    for i, token_name in enumerate(SPECIAL_TOKENS):
        special_tokens[token_name] = tokens_offset + i

    # 创建新的 encoding，包含所有 token
    enc = tiktoken.Encoding(
        name=f"{encoding_name}_nanochat",
        pat_str=base_enc._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    print(f"Final vocab size: {enc.n_vocab} (+{len(SPECIAL_TOKENS)} special tokens)")

    # 2. 保存 tokenizer.pkl
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved tokenizer to {pickle_path}")

    # 3. 生成 token_bytes.pt
    # 这个文件记录每个 token 对应多少 UTF-8 字节，用于计算 bits per byte 指标
    print("Generating token_bytes.pt...")
    vocab_size = enc.n_vocab
    special_set = enc.special_tokens_set
    token_bytes = []

    for token_id in range(vocab_size):
        try:
            token_str = enc.decode([token_id])
            if token_str in special_set:
                token_bytes.append(0)  # 特殊 token 不计入
            else:
                id_bytes = len(token_str.encode("utf-8"))
                token_bytes.append(id_bytes)
        except Exception:
            token_bytes.append(0)  # 无法解码的 token

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # 统计信息
    token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
    print("\nToken bytes stats (excluding special tokens):")
    print(f"  Min: {int(token_bytes_nonzero.min().item())}")
    print(f"  Max: {int(token_bytes_nonzero.max().item())}")
    print(f"  Mean: {token_bytes_nonzero.mean().item():.2f}")

    print("\n" + "=" * 50)
    print(f"Done! {friendly_name} tokenizer is ready.")
    print("Files created:")
    print(f"  - {pickle_path}")
    print(f"  - {token_bytes_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
