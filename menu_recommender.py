"""RAG-style recommender over the Berkeley dining menu JSON."""

from __future__ import annotations

import json
import math
import hashlib
from joblib import dump, load
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_ROOT = Path(".cache") / "recommender"
DOC_TEXT_MAX_LEN = 4000


@dataclass
class MenuItem:
    """Flattened view of a single menu item for retrieval."""

    name: str
    location: str
    meal: str
    category: str
    tags: List[str]
    tag_set: Set[str]
    icons: List[str]
    dietary_choices: List[str]
    nutrition: Dict[str, float]
    ingredients: Optional[str]
    notes: Optional[str]
    menu_reference: Dict[str, Any]
    hours: List[str]
    hours_structured: List[Dict[str, Any]]
    doc_text: str
    source_payload: Dict[str, Any]


class MenuKnowledgeBase:
    """Loads menu.json and prepares menu items for retrieval."""

    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        self.raw_payload: Dict[str, Any] = {}
        self.items: List[MenuItem] = []
        self.fingerprint: Optional[str] = None

    def load(self) -> None:
        raw_bytes = self.json_path.read_bytes()
        self.fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        self.raw_payload = json.loads(raw_bytes)
        self.items = list(self._iter_items(self.raw_payload))

    def _iter_items(self, payload: Dict[str, Any]) -> Iterable[MenuItem]:
        locations = payload.get("locations", [])
        icon_lookup = payload.get("icon_legend", {})
        for location in locations:
            loc_name = location.get("name") or "Unknown Location"
            hours_display = location.get("hours", [])
            hours_structured = location.get("hours_structured", [])
            for meal in location.get("meals", []):
                meal_label = meal.get("label") or "Meal"
                for category in meal.get("categories", []):
                    category_name = category.get("category") or "General"
                    for item in category.get("items", []):
                        nutrition = self._normalize_nutrition(item.get("nutrition", {}))
                        dietary_choices = sorted(
                            choice
                            for choice, allowed in (item.get("dietary_choices") or {}).items()
                            if allowed
                        )
                        icon_labels = [
                            icon.get("label")
                            for icon in item.get("icons", [])
                            if icon.get("label")
                        ]
                        coded_icons = [
                            icon_lookup.get(icon.get("code") or "", {}).get("label")
                            for icon in item.get("icons", [])
                        ]
                        combined_icons = sorted(
                            {label for label in icon_labels + coded_icons if label}
                        )
                        tag_list = item.get("tags", [])
                        tag_set = set(tag_list)
                        doc_text = self._build_doc_text(
                            item=item,
                            location=loc_name,
                            meal=meal_label,
                            category=category_name,
                            tags=tag_list,
                            icons=combined_icons,
                            dietary=dietary_choices,
                        )
                        yield MenuItem(
                            name=item.get("name") or "Unknown Item",
                            location=loc_name,
                            meal=meal_label,
                            category=category_name,
                            tags=tag_list,
                            tag_set=tag_set,
                            icons=combined_icons,
                            dietary_choices=dietary_choices,
                            nutrition=nutrition,
                            ingredients=item.get("ingredients"),
                            notes=item.get("notes"),
                            menu_reference=item.get("menu_reference", {}),
                            hours=hours_display,
                            hours_structured=hours_structured,
                            doc_text=doc_text,
                            source_payload=item,
                        )

    @staticmethod
    def _normalize_nutrition(raw: Dict[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for key, value in raw.items():
            if value in (None, "", "N/A"):
                continue
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _build_doc_text(
        item: Dict[str, Any],
        location: str,
        meal: str,
        category: str,
        tags: Sequence[str],
        icons: Sequence[str],
        dietary: Sequence[str],
    ) -> str:
        parts = [
            item.get("name") or "",
            item.get("short_name") or "",
            item.get("recipe_description") or "",
            " ".join(tags),
            " ".join(icons),
            " ".join(dietary),
            item.get("ingredients") or "",
            item.get("notes") or "",
        ]
        nutrition = item.get("nutrition") or {}
        macros = " ".join(
            f"{key} {value}"
            for key, value in nutrition.items()
            if value not in (None, "", "N/A")
        )
        parts.extend(
            [
                f"Location: {location}",
                f"Meal: {meal}",
                f"Category: {category}",
                macros,
            ]
        )
        text = " ".join(str(part) for part in parts if part).strip()
        if len(text) > DOC_TEXT_MAX_LEN:
            text = text[:DOC_TEXT_MAX_LEN]
        return text


class MenuRecommender:
    """Retrieval augmented generator for menu recommendations."""

    def __init__(self, kb: MenuKnowledgeBase, min_protein: float = 20.0) -> None:
        self.kb = kb
        self.min_protein = min_protein
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_matrix = None

    def build_index(self) -> None:
        documents = [item.doc_text for item in self.kb.items]
        if not documents:
            raise ValueError("Knowledge base is empty. Did you load menus.json?")
        self.document_matrix = self.vectorizer.fit_transform(documents)

    def save_index(self, cache_path: Path) -> None:
        if self.document_matrix is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "fingerprint": self.kb.fingerprint,
                "vectorizer": self.vectorizer,
                "matrix": self.document_matrix,
            },
            cache_path,
        )

    def load_index(self, cache_path: Path) -> bool:
        if not cache_path.exists() or self.kb.fingerprint is None:
            return False
        try:
            payload = load(cache_path)
        except Exception:  # noqa: BLE001
            return False
        if payload.get("fingerprint") != self.kb.fingerprint:
            return False
        self.vectorizer = payload.get("vectorizer", self.vectorizer)
        self.document_matrix = payload.get("matrix")
        return self.document_matrix is not None

    def recommend(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.document_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vec = self.vectorizer.transform([query])
        scores = (self.document_matrix @ query_vec.T).toarray().ravel()
        ranked_indices = np.argsort(scores)[::-1]

        hits: List[Tuple[float, MenuItem]] = []
        for idx in ranked_indices:
            base_score = scores[idx]
            if base_score <= 0:
                break
            item = self.kb.items[idx]
            score = self._apply_heuristics(item, query, base_score)
            hits.append((score, item))

        hits.sort(key=lambda pair: pair[0], reverse=True)
        top_hits = hits[:top_k]
        return [self._format_result(item, score) for score, item in top_hits]

    def _apply_heuristics(self, item: MenuItem, query: str, base_score: float) -> float:
        score = base_score
        normalized = query.lower()

        protein = item.nutrition.get("Protein (g)")
        if protein:
            if "protein" in normalized:
                score += math.log1p(protein)
            if protein >= self.min_protein:
                score += 0.5

        calories = item.nutrition.get("Calories (kcal)")
        if calories:
            if "low calorie" in normalized or "light" in normalized:
                score += 0.5 / math.log1p(calories)
            if "high calorie" in normalized:
                score += math.log1p(calories)

        carbs = item.nutrition.get("Carbohydrate (g)")
        if carbs is not None:
            if "low carb" in normalized:
                score += 1 / (1 + carbs)
            if "carb" in normalized and carbs > 0:
                score += math.log1p(carbs)

        fat = item.nutrition.get("Total Lipid/Fat (g)")
        if fat is not None and "fat" in normalized:
            score += math.log1p(fat)

        for keyword, tag_list in [
            ("vegan", ["vegan-option"]),
            ("vegetarian", ["vegetarian-option"]),
            ("gluten free", ["gluten"]),
            ("halal", ["halal"]),
            ("kosher", ["kosher"]),
        ]:
            if keyword in normalized:
                if any(tag in item.tag_set for tag in tag_list):
                    score += 1.5
                if keyword == "gluten free" and "Gluten" in item.dietary_choices:
                    score += 1.5

        if "spicy" in normalized and any("spicy" in tag for tag in item.tag_set):
            score += 0.75

        if "dessert" in normalized and item.category.lower().startswith("dessert"):
            score += 0.75

        return score

    def _format_result(self, item: MenuItem, score: float) -> Dict[str, Any]:
        nutrition_summary = self._build_nutrition_summary(item.nutrition)
        description = self._build_description(item, nutrition_summary)
        return {
            "name": item.name,
            "location": item.location,
            "meal": item.meal,
            "category": item.category,
            "score": round(float(score), 4),
            "tags": item.tags,
            "dietary_choices": item.dietary_choices,
            "nutrition": nutrition_summary,
            "ingredients": item.ingredients,
            "notes": item.notes,
            "hours": item.hours,
            "hours_structured": item.hours_structured,
            "blurb": description,
            "context": item.doc_text,
            "menu_reference": item.menu_reference,
        }

    @staticmethod
    def _build_nutrition_summary(nutrition: Dict[str, float]) -> Dict[str, float]:
        keys = [
            "Calories (kcal)",
            "Protein (g)",
            "Carbohydrate (g)",
            "Total Lipid/Fat (g)",
            "Total Dietary Fiber (g)",
            "Sodium (mg)",
        ]
        return {
            key: round(nutrition[key], 2)
            for key in keys
            if key in nutrition
        }

    @staticmethod
    def _build_description(item: MenuItem, nutrition_summary: Dict[str, float]) -> str:
        pieces = [
            f"{item.name} at {item.location} ({item.meal})",
        ]
        if item.category:
            pieces.append(f"Category: {item.category}.")

        if item.dietary_choices:
            choices = ", ".join(item.dietary_choices)
            pieces.append(f"Dietary: {choices}.")

        if nutrition_summary:
            macros = ", ".join(f"{key}: {value}" for key, value in nutrition_summary.items())
            pieces.append(f"Key nutrition: {macros}.")

        if item.ingredients:
            ingredients = item.ingredients.replace("\n", " ")
            pieces.append(f"Ingredients: {ingredients}")

        if item.notes:
            pieces.append(f"Notes: {item.notes}")

        return " ".join(pieces)


def build_recommender(json_path: Path) -> MenuRecommender:
    kb = MenuKnowledgeBase(json_path)
    kb.load()
    recommender = MenuRecommender(kb)
    cache_path = CACHE_ROOT / f"{json_path.stem}-{kb.fingerprint[:16]}.joblib"
    if not recommender.load_index(cache_path):
        recommender.build_index()
        recommender.save_index(cache_path)
    return recommender


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Query the dining hall recommender.")
    parser.add_argument("query", help="What are you craving?")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to return (default 5).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("menus.json"),
        help="Path to the menus JSON file.",
    )
    args = parser.parse_args()

    recommender = build_recommender(args.json)
    results = recommender.recommend(args.query, top_k=args.top_k)
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result['blurb']}")


if __name__ == "__main__":
    main()
