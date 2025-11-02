import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser(description="MedTokAlign - Tokenizer Adaptation Runner (TokenAlign pipeline)")
    ap.add_argument("--model_id", type=str, required=True, help="Base model id or path")
    ap.add_argument("--top_k", type=int, default=8192, help="Number of new medical tokens to add to target tokenizer")
    ap.add_argument("--pivot", type=int, default=300, help="Number of pivot tokens for relative representation")
    args = ap.parse_args()

    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_vocab_adaptation.sh")
    cmd = [script, "--model_id", args.model_id, "--top_k", str(args.top_k), "--pivot", str(args.pivot)]
    print(f"[MedTokAlign] Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()


