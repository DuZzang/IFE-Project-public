"""Explanatory figures of the I20 chart images the CNN reads (for the team report).

Picks, for one out-of-sample formation month, three stocks the ensemble ranked in decile 10
(highest predicted chance of going up) and three in decile 1, and draws their official 64 x 60
images with the ensemble score and the realized label (next month up or down). Stocks are not
identified. Images come from the authors' public research data: team-internal use only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cnn_official_data import (  # noqa: E402
    catalog_archive,
    iter_filtered_year_image_batches,
    read_year_labels,
    validate_label_partition,
)

IMAGE_ZIP = ROOT / "handoff/금공개 팀플/CNN/img_data.zip"
SCORES = ROOT / "outputs/cnn_oos_deciles_2001_2019/cnn_signal_scores_2001_2019.csv.gz"
OUTPUT_DIR = ROOT / "outputs/cnn_image_examples"
MONTH, YEAR, SEED = "2019-06", 2019, 11
TEXT = {
    "ko": {
        "font": "Malgun Gothic",
        "anatomy_title": "CNN이 보는 입력: 20거래일 차트 이미지 (64 × 60 픽셀, 흑백)",
        "price": "가격 영역\n하루마다 막대 하나:\n고가~저가 세로선,\n왼쪽 눈금 = 시가,\n오른쪽 눈금 = 종가\n+ 20일 이동평균선",
        "volume": "거래량 영역\n(하루 거래량 막대)",
        "days": "가로 60픽셀 = 20거래일 × 하루 3픽셀",
        "last": "마지막 날 = 월말\n(점수 계산 시점)",
        "examples_title": "같은 달(2019년 6월 말) 모델이 매긴 점수와 실제 결과",
        "high": "상승확률 높음 (10분위)",
        "low": "상승확률 낮음 (1분위)",
        "score": "모델 점수 P(상승) = {p:.2f}",
        "up": "정답: 다음 달 상승",
        "down": "정답: 다음 달 하락",
        "note": "정답(라벨) = 이미지 마지막 날(월말)부터 다음 월말까지 수익률이 0보다 크면 1, 아니면 0. "
                "학습(1993~2000)에서만 쓰고, 2001년 이후에는 점수만 씁니다. 종목은 표시하지 않았습니다.",
    },
    "en": {
        "font": "DejaVu Sans",
        "anatomy_title": "What the CNN reads: a 20-trading-day chart image (64 × 60 pixels, black and white)",
        "price": "Price area\none bar per day:\nhigh-low vertical line,\nleft tick = open,\nright tick = close\n+ 20-day moving average",
        "volume": "Volume area\n(daily volume bars)",
        "days": "60 pixels wide = 20 trading days × 3 pixels per day",
        "last": "Last day = month-end\n(when the score is computed)",
        "examples_title": "Scores the model gave in the same month (end of June 2019) and what actually happened",
        "high": "High chance of going up (decile 10)",
        "low": "Low chance of going up (decile 1)",
        "score": "Model score P(up) = {p:.2f}",
        "up": "Label: up next month",
        "down": "Label: down next month",
        "note": "Label = 1 if the return from the image's last day (month-end) to the next month-end is above 0, else 0. "
                "Used only for training (1993-2000); from 2001 on only the score is used. Stocks are not identified.",
    },
}


def load_examples() -> tuple[list[dict], np.ndarray]:
    scores = pd.read_csv(SCORES, usecols=["PERMNO", "formation_month", "cnn_prob_up", "cnn_decile"], dtype={"formation_month": str})
    month = scores.loc[scores["formation_month"] == MONTH]
    catalog = catalog_archive(IMAGE_ZIP)
    partition = next(item for item in catalog if item.year == YEAR)
    _, labels = validate_label_partition(partition, read_year_labels(IMAGE_ZIP, YEAR, catalog=catalog))
    labels = labels.assign(row=np.arange(len(labels)))
    merged = month.merge(labels[["PERMNO", "month", "Ret_month", "row"]].dropna(subset=["Ret_month"]),
                         left_on=["PERMNO", "formation_month"], right_on=["PERMNO", "month"], how="inner")
    rng = np.random.default_rng(SEED)
    picks = []
    for decile in [10, 1]:
        pool = merged.loc[merged["cnn_decile"] == decile]
        picks.append(pool.iloc[rng.choice(len(pool), size=3, replace=False)])
    chosen = pd.concat(picks)
    mask = np.zeros(partition.image_count, dtype=bool)
    mask[chosen["row"].to_numpy()] = True
    images = {}
    for rows, batch in iter_filtered_year_image_batches(IMAGE_ZIP, YEAR, mask, batch_size=4096, catalog=catalog):
        for row, image in zip(rows, batch):
            images[int(row)] = np.array(image)
    records = [
        {"decile": int(r.cnn_decile), "score": float(r.cnn_prob_up), "up": bool(r.Ret_month > 0), "image": images[int(r.row)]}
        for r in chosen.itertuples()
    ]
    stack = np.stack([r["image"] for r in records])
    return records, stack


def volume_split(stack: np.ndarray) -> int:
    """First row of the volume panel: the row after the blank gap in the lower part of the image."""
    used = (stack > 0).any(axis=(0, 2))
    lower = np.arange(len(used)) > len(used) // 2
    blank = np.flatnonzero(~used & lower)
    return int(blank.max()) + 1 if len(blank) else int(len(used) * 0.8)


def draw_anatomy(record: dict, split: int, lang: str) -> Path:
    t = TEXT[lang]
    image = record["image"]
    with plt.rc_context({"font.family": t["font"], "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(11, 6.2))
        ax = fig.add_axes([0.05, 0.12, 0.42, 0.74])
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#9CA3AF")
        ax.axhline(split - 0.5, color="#F59E0B", linewidth=1.2, linestyle="--")
        ax.axvline(56.5, color="#60A5FA", linewidth=1.2, linestyle=":")

        def note(text: str, xy: tuple[float, float], xytext: tuple[float, float]) -> None:
            fig.text(*xytext, text, fontsize=10, color="#111827", va="center", ha="left", linespacing=1.35)
            arrow = FancyArrowPatch(posA=fig.transFigure.inverted().transform(ax.transData.transform(xy)),
                                    posB=(xytext[0] - 0.01, xytext[1]), transform=fig.transFigure,
                                    arrowstyle="-", color="#6B7280", linewidth=0.9)
            fig.patches.append(arrow)

        note(t["price"], (60, split * 0.35), (0.52, 0.66))
        note(t["volume"], (60, (split + 64) / 2), (0.52, 0.27))
        note(t["last"], (58, -1), (0.52, 0.92))
        fig.text(0.05, 0.06, t["days"], fontsize=10, color="#374151")
        fig.suptitle(t["anatomy_title"], fontsize=13.5, fontweight="bold", x=0.05, ha="left", y=0.99)
        output = OUTPUT_DIR / f"cnn_image_anatomy_{lang}.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def draw_examples(records: list[dict], lang: str) -> Path:
    t = TEXT[lang]
    with plt.rc_context({"font.family": t["font"], "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.8))
        for row_index, (decile, label) in enumerate([(10, t["high"]), (1, t["low"])]):
            group = [r for r in records if r["decile"] == decile]
            for col, record in enumerate(group):
                ax = axes[row_index, col]
                ax.imshow(record["image"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                outcome = t["up"] if record["up"] else t["down"]
                color = "#1F6F43" if record["up"] else "#A6261B"
                ax.set_title(t["score"].format(p=record["score"]), fontsize=10, pad=4)
                ax.set_xlabel(outcome, fontsize=10, color=color)
            axes[row_index, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        fig.suptitle(t["examples_title"], fontsize=13.5, fontweight="bold", x=0.03, ha="left", y=0.985)
        fig.text(0.03, 0.02, t["note"], fontsize=8.8, color="#374151", wrap=True)
        fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.95), h_pad=3.0)
        output = OUTPUT_DIR / f"cnn_image_examples_{lang}.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, stack = load_examples()
    split = volume_split(stack)
    print("image shape:", stack.shape[1:], "volume panel starts at row:", split)
    print("scores:", [(r["decile"], round(r["score"], 3), r["up"]) for r in records])
    for lang in ["ko", "en"]:
        print(draw_anatomy(records[0], split, lang))
        print(draw_examples(records, lang))


if __name__ == "__main__":
    main()
