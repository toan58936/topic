"""CLI dự đoán review dùng đúng model/checkpoint của project.

Ví dụ:
  python -m src.predict_cli --checkpoint checkpoints/bert/best_model.pt --text "Great food and service"
  python -m src.predict_cli --checkpoint checkpoints/bert/best_model.pt --input data/reviews.csv --text-column text --output results/review_predictions.csv
"""
import argparse
from pathlib import Path
import pandas as pd
from src.predict import load_trained_model, predict
from src.model_builder import build_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Predict Yelp review stars")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn best_model.pt")
    parser.add_argument("--text", help="Một review cần dự đoán")
    parser.add_argument("--input", help="CSV/JSON/JSONL chứa danh sách review")
    parser.add_argument("--text-column", default="text", help="Tên cột chứa nội dung review")
    parser.add_argument("--output", default="results/predictions.csv", help="File CSV đầu ra")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if bool(args.text) == bool(args.input):
        parser.error("Chỉ chọn một trong --text hoặc --input")

    model, info = load_trained_model(args.checkpoint, device=args.device)
    tokenizer = build_tokenizer(info["backbone"])
    max_length = int(info["run_config"].get("max_length", 256))

    if args.text:
        score = predict(args.text, model, tokenizer, device=args.device,
                        max_length=max_length, batch_size=args.batch_size)
        print(f"Review: {args.text}\nĐiểm dự đoán: {score:.3f}/5")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(input_path)
    elif suffix in (".jsonl", ".ndjson"):
        frame = pd.read_json(input_path, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(input_path)
    else:
        raise ValueError("File đầu vào phải là CSV, JSON hoặc JSONL")
    if args.text_column not in frame.columns:
        raise ValueError(f"Không có cột {args.text_column!r}; cột hiện có: {list(frame.columns)}")
    texts = frame[args.text_column].fillna("").astype(str).tolist()
    scores = predict(texts, model, tokenizer, device=args.device,
                     max_length=max_length, batch_size=args.batch_size)
    frame["predicted_stars"] = scores
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã dự đoán {len(frame)} review. File kết quả: {output}")


if __name__ == "__main__":
    main()
