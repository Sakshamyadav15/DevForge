#!/usr/bin/env python3
"""Generate diverse general-life documents for DevForge snapshots."""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "general_topics.json"
DEFAULT_DOCUMENTS_PER_TOPIC = 100

@dataclass(frozen=True)
class Topic:
    name: str
    category: str

TOPICS: List[Topic] = [
    Topic("Breakfast recipes", "Food & Cooking"),
    Topic("Quick dinner ideas", "Food & Cooking"),
    Topic("Healthy snacks", "Food & Cooking"),
    Topic("Meal prep tips", "Food & Cooking"),
    Topic("Coffee & beverages", "Food & Cooking"),
    Topic("Home organization", "Home & Living"),
    Topic("Cleaning hacks", "Home & Living"),
    Topic("Interior decorating", "Home & Living"),
    Topic("Gardening basics", "Home & Living"),
    Topic("DIY home repairs", "Home & Living"),
    Topic("Morning routines", "Health & Wellness"),
    Topic("Sleep improvement", "Health & Wellness"),
    Topic("Stress management", "Health & Wellness"),
    Topic("Exercise at home", "Health & Wellness"),
    Topic("Nutrition basics", "Health & Wellness"),
    Topic("Budgeting tips", "Personal Finance"),
    Topic("Saving money", "Personal Finance"),
    Topic("Online shopping deals", "Personal Finance"),
    Topic("Side income ideas", "Personal Finance"),
    Topic("Investing basics", "Personal Finance"),
    Topic("Dog care", "Animals & Pets"),
    Topic("Cat care", "Animals & Pets"),
    Topic("Wildlife facts", "Animals & Pets"),
    Topic("Marine animals", "Animals & Pets"),
    Topic("Bird watching", "Animals & Pets"),
    Topic("Indoor plants", "Plants & Nature"),
    Topic("Vegetable gardening", "Plants & Nature"),
    Topic("Flower types", "Plants & Nature"),
    Topic("Tree identification", "Plants & Nature"),
    Topic("Plant care tips", "Plants & Nature"),
    Topic("Car maintenance", "Vehicles & Transport"),
    Topic("Buying a car", "Vehicles & Transport"),
    Topic("Motorcycles", "Vehicles & Transport"),
    Topic("Electric vehicles", "Vehicles & Transport"),
    Topic("Public transport tips", "Vehicles & Transport"),
    Topic("Study techniques", "Education"),
    Topic("College admissions", "Education"),
    Topic("Online learning", "Education"),
    Topic("Career planning", "Education"),
    Topic("Student life tips", "Education"),
    Topic("Movie recommendations", "Entertainment"),
    Topic("Book suggestions", "Entertainment"),
    Topic("Music genres", "Entertainment"),
    Topic("Video games", "Entertainment"),
    Topic("Hobbies to try", "Entertainment"),
    Topic("Smartphone tips", "Tech & Daily Life"),
    Topic("Social media usage", "Tech & Daily Life"),
    Topic("Useful apps", "Tech & Daily Life"),
    Topic("Online safety", "Tech & Daily Life"),
    Topic("Travel planning", "Tech & Daily Life"),
]

INTRO_TEMPLATES = [
    "People often ask about {topic} because it affects daily routines.",
    "Here is a practical walkthrough of {topic} that anyone can follow.",
    "Understanding {topic} helps simplify everyday decisions.",
    "This guide explores {topic} with real-world examples.",
]

DETAIL_TEMPLATES = [
    "Start with the essentials: {tip_one}. Keep an eye on {tip_two} to stay consistent.",
    "Break the process into three parts: planning, doing, and reviewing progress.",
    "Most beginners overlook {tip_three}, yet it makes a noticeable difference.",
    "Pair this habit with a simple tracking system to notice patterns over time.",
    "A quick checklist: {tip_one}, {tip_two}, and a reminder to adjust weekly.",
]

ACTION_TEMPLATES = [
    "Actionable idea #{n}: dedicate {minutes} minutes to this task and log the outcome.",
    "Try setting a weekly experiment focused on {focus_area} and journal what changes.",
    "Share your progress with a friend or online group to stay accountable.",
]

SUMMARY_TEMPLATES = [
    "In short, consistent attention to {topic} builds confidence in daily life.",
    "Small improvements in {topic} compound quickly when reviewed every Friday.",
    "Treat {topic} like a mini-project: define goals, test ideas, and celebrate wins.",
]

SAMPLE_TIPS = [
    "moderate planning blocks",
    "simple ingredient swaps",
    "budget-friendly alternatives",
    "low-effort cleanup routines",
    "weekly reflection notes",
    "seasonal adjustments",
    "time-boxed practice sessions",
]

FOCUS_AREAS = [
    "energy",
    "budget",
    "family time",
    "creativity",
    "skill-building",
    "wellness",
    "organization",
]


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return cleaned.strip("-")


def build_text(rng: random.Random, topic: str) -> str:
    intro = rng.choice(INTRO_TEMPLATES).format(topic=topic)
    detail = rng.choice(DETAIL_TEMPLATES).format(
        tip_one=rng.choice(SAMPLE_TIPS),
        tip_two=rng.choice(SAMPLE_TIPS),
        tip_three=rng.choice(SAMPLE_TIPS),
    )
    action = rng.choice(ACTION_TEMPLATES).format(
        n=rng.randint(1, 3),
        minutes=rng.choice([10, 15, 20, 25]),
        focus_area=rng.choice(FOCUS_AREAS),
    )
    summary = rng.choice(SUMMARY_TEMPLATES).format(topic=topic)
    return "\n\n".join([intro, detail, action, summary])


def create_document(topic: Topic, index: int, rng: random.Random) -> dict:
    slug = slugify(topic.name)
    doc_id = f"general_{slug}_{index:03d}"
    title = f"{topic.name}: everyday idea #{index + 1}"
    text = build_text(rng, topic.name)
    metadata = {
        "topic": topic.name,
        "category": topic.category,
        "topic_slug": slug,
        "order": index + 1,
        "created_at": datetime.utcnow().isoformat(),
        "source": "synthetic_general_topics_v1",
    }
    return {"id": doc_id, "title": title, "text": text, "metadata": metadata}


def generate_documents(per_topic: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    documents: list[dict] = []
    for topic in TOPICS:
        for idx in range(per_topic):
            documents.append(create_document(topic, idx, rng))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate general-life documents for DevForge")
    parser.add_argument("--per-topic", type=int, default=DEFAULT_DOCUMENTS_PER_TOPIC,
                        help="Number of documents per topic (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON file path (default: backend/data/general_topics.json)")

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(TOPICS)} topics × {args.per_topic} docs each...")
    documents = generate_documents(args.per_topic, args.seed)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(documents):,} documents to {args.output}")
    print("Next: ingest via scripts/ingest_bulk.py --source json --file <output>")


if __name__ == "__main__":
    main()
