from transformers import AutoConfig
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="google/gemma-2b")
    args = parser.parse_args()
    cfg = AutoConfig.from_pretrained(args.model_path)
    print(getattr(cfg, "vocab_size", 0))


