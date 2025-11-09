import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import pandas as pd
import numpy as np

LABELS = ["left", "center", "right"]

def predict_bias(texts, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = [LABELS[np.argmax(p)] for p in probs]

    return preds, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for news bias classification")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input", type=str, required=True, help="Text or CSV path")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV")

    args = parser.parse_args()

    # Check if input is text or file
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        preds, probs = predict_bias(df["text"].tolist(), args.model_dir)
        df["predicted_bias"] = preds
        df["prob_left"] = probs[:, 0]
        df["prob_center"] = probs[:, 1]
        df["prob_right"] = probs[:, 2]
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print(df.head())
    else:
        # Direct text input
        preds, probs = predict_bias([args.input], args.model_dir)
        print(f"\nText: {args.input[:100]}...")
        print(f"Predicted Bias: {preds[0]} (L={probs[0][0]:.2f}, C={probs[0][1]:.2f}, R={probs[0][2]:.2f})")
