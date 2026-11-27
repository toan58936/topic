"""CLI dự đoán review dùng đúng model/checkpoint của project.

Ví dụ:
  python -m src.predict_cli --checkpoint checkpoints/bert/best_model.pt --text "Great food and service"
  python -m src.predict_cli --checkpoint checkpoints/roberta_v2/best_model.pt --input data/reviews.csv --text-column text --output results/review_predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.model_builder import build_tokenizer
from src.predict import load_trained_model, predict
from src.utils import write_atomic


def main():
    parser = argparse.ArgumentParser(description="Predict Yelp review stars")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn best_model.pt")
    parser.add_argument("--text", default=None, help="Một review cần dự đoán")
    parser.add_argument("--input", default=None, help="CSV/JSON/JSONL chứa danh sách review")
    parser.add_argument("--text-column", default="text", help="Tên cột chứa nội dung review")
    parser.add_argument("--output", default="results/predictions.csv", help="File CSV đầu ra")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if (args.text is not None) == (args.input is not None):
        parser.error("Chỉ chọn một trong --text hoặc --input")

    model, info = load_trained_model(args.checkpoint, device=args.device)
    tokenizer = build_tokenizer(info["backbone"])
    run_cfg = info.get("run_config", {})
    max_length = int(run_cfg.get("max_length", 256))
    truncation_strategy = run_cfg.get("truncation_strategy", "tail")

    if args.text is not None:
        if not args.text.strip():
            raise ValueError("--text không được rỗng")
        score = predict(
            args.text,
            model,
            tokenizer,
            device=args.device,
            max_length=max_length,
            batch_size=args.batch_size,
            truncation_strategy=truncation_strategy,
        )
        print(
            f"Model: {info['backbone']} (max_length={max_length}, strategy={truncation_strategy})\n"
            f"Review: {args.text}\n"
            f"Điểm dự đoán: {score:.3f}/5"
        )
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
        raise ValueError(
            f"Không có cột {args.text_column!r}; cột hiện có: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"File đầu vào rỗng, không có review để dự đoán: {input_path}")

    texts = frame[args.text_column].fillna("").astype(str).tolist()
    scores = predict(
        texts,
        model,
        tokenizer,
        device=args.device,
        max_length=max_length,
        batch_size=args.batch_size,
        truncation_strategy=truncation_strategy,
    )
    frame["predicted_stars"] = scores

    output = Path(args.output)
    write_atomic(
        str(output),
        lambda p: frame.to_csv(p, index=False, encoding="utf-8-sig"),
    )
    print(
        f"Đã dự đoán {len(frame)} review bằng {info['backbone']} "
        f"(max_length={max_length}, strategy={truncation_strategy}). "
        f"File kết quả: {output}"
    )


if __name__ == "__main__":
    main()