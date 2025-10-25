"""RAG-style recommender over the Berkeley dining menu JSON."""

from __future__ import annotations

import json
import math
import hashlib
from joblib import dump, load
from dataclasses import dataclass
from pathlib import Path
import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from crowdedness import estimate_crowdedness

CACHE_ROOT = Path(".cache") / "recommender"
DOC_TEXT_MAX_LEN = 4000
MEAT_KEYWORDS = {
    "beef",
    "chicken",
    "turkey",
    "pork",
    "bacon",
    "ham",
    "lamb",
    "fish",
    "shrimp",
    "tuna",
    "salmon",
    "sausage",
}
DESSERT_CATEGORIES = {
    "dessert",
    "bakery",
    "pastries",
    "pastry",
    "sweets",
    "breakfast pastries",
    "cake",
    "ice cream",
}
KNOWN_LOCATIONS = {
    "cafe 3": "Cafe 3",
    "crossroads": "Crossroads",
    "foothill": "Foothill",
    "clark kerr": "Clark Kerr Campus",
    "clark kerr campus": "Clark Kerr Campus",
    "golden bear cafe": "Golden Bear Cafe",
    "golden bear": "Golden Bear Cafe",
    "browns cafe": "Browns Cafe",
    "bear market": "Bear Market",
    "eateries at the student union": "Eateries At The Student Union",
    "local x design": "Local X Design",
    "den": "Den",
}


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
    status: Optional[str]
    crowdedness: Optional[Dict[str, Any]]
    service_windows: List[Dict[str, Any]]
    doc_text: str
    source_payload: Dict[str, Any]


@dataclass
class RequestProfile:
    """Normalized view of what the user actually requested."""

    normalized_query: str
    vegetarian: bool = False
    vegan: bool = False
    gluten_free: bool = False
    halal: bool = False
    kosher: bool = False
    dairy_free: bool = False
    low_carbon: bool = False
    healthy_focus: bool = False

    @property
    def any_classification_requested(self) -> bool:
        return any(
            [
                self.vegetarian,
                self.vegan,
                self.gluten_free,
                self.halal,
                self.kosher,
                self.dairy_free,
                self.low_carbon,
            ]
        )

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
        items = list(self._iter_items(self.raw_payload))
        timestamp = dt.datetime.now(dt.timezone.utc)
        for item in items:
            crowd = estimate_crowdedness(item.location, when=timestamp)
            item.crowdedness = {
                "score": crowd.score,
                "label": crowd.label,
                "updated_at": crowd.updated_at,
                "source": crowd.source,
            }
        self.items = items

    def _iter_items(self, payload: Dict[str, Any]) -> Iterable[MenuItem]:
        locations = payload.get("locations", [])
        icon_lookup = payload.get("icon_legend", {})
        for location in locations:
            loc_name = location.get("name") or "Unknown Location"
            hours_display = location.get("hours", [])
            hours_structured = location.get("hours_structured", [])
            status = location.get("status")
            for meal in location.get("meals", []):
                meal_label = meal.get("label") or "Meal"
                service_windows = meal.get("schedule") or []
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
                            status=status,
                            crowdedness=None,
                            service_windows=service_windows,
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

        augmented_query = self._augment_query_terms(query)
        time_hint = self._extract_time_hints(augmented_query)
        locations_filter = self._extract_locations(augmented_query)
        normalized_query = augmented_query.lower()
        request = self._build_request_profile(normalized_query)
        query_vec = self.vectorizer.transform([augmented_query])
        scores = (self.document_matrix @ query_vec.T).toarray().ravel()
        ranked_indices = np.argsort(scores)[::-1]

        hits: List[Tuple[float, MenuItem]] = []
        now = dt.datetime.now(dt.timezone.utc)
        for idx in ranked_indices:
            base_score = scores[idx]
            if base_score <= 0:
                break
            item = self.kb.items[idx]
            if not self._is_serving_now(item, now, time_hint):
                continue
            if locations_filter and item.location not in locations_filter:
                continue
            score = self._apply_heuristics(item, request, base_score)
            hits.append((score, item))

        hits.sort(key=lambda pair: pair[0], reverse=True)
        filtered: List[Tuple[float, MenuItem]] = []
        for score, item in hits:
            if not self._passes_filters(item, request):
                continue
            filtered.append((score, item))
            if len(filtered) >= top_k:
                break
        return [self._format_result(item, score, request) for score, item in filtered]

    def _apply_heuristics(self, item: MenuItem, request: RequestProfile, base_score: float) -> float:
        normalized = request.normalized_query
        score = base_score

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

        crowd_adjust = False
        if any(word in normalized for word in ["quick", "fast", "short wait", "not crowded", "less busy", "wait time"]):
            crowd_adjust = True
        elif "busy" in normalized or "crowded" in normalized:
            crowd_adjust = True

        icons_lower = {icon.lower() for icon in item.icons}
        dietary_lower = {choice.lower() for choice in item.dietary_choices}

        if request.vegan and "vegan-option" in item.tag_set:
            score += 1.5
        if request.vegetarian and (
            "vegetarian-option" in item.tag_set or "vegan-option" in item.tag_set
        ):
            score += 1.5
        if request.gluten_free:
            if "gluten" in item.tag_set:
                score -= 1.0
            if "gluten" in dietary_lower or "gluten free" in dietary_lower:
                score += 1.5
        if request.halal and "halal" in item.tag_set:
            score += 1.25
        if request.kosher and "kosher" in item.tag_set:
            score += 1.25
        if request.dairy_free:
            if "milk" in item.tag_set:
                score -= 1.0
            else:
                score += 0.25
        if request.low_carbon and "low carbon footprint" in icons_lower:
            score += 0.75

        if "spicy" in normalized and any("spicy" in tag for tag in item.tag_set):
            score += 0.75

        if "dessert" in normalized and item.category.lower().startswith("dessert"):
            score += 0.75

        crowd = item.crowdedness or {}
        crowd_label = str(crowd.get("label", "")).lower()
        crowd_score = crowd.get("score")
        if any(word in normalized for word in ["quick", "fast", "short wait", "not crowded", "less busy"]):
            if crowd_label == "low":
                score += 1.0
            elif crowd_label == "medium":
                score += 0.25
            elif crowd_label == "high":
                score -= 0.75
        elif "busy" in normalized and crowd_label == "high":
            score += 0.5

        if crowd_score is not None and crowd_adjust:
            score += (crowd_score - 0.5) * 0.2

        score += self._location_boost(normalized, item.location)
        score += self._nutrition_priority_boost(item, request)

        return score

    def _passes_filters(self, item: MenuItem, request: RequestProfile) -> bool:
        normalized = request.normalized_query
        icons_lower = {icon.lower() for icon in item.icons}

        if request.vegetarian and "vegetarian-option" not in item.tag_set:
            if "vegan-option" not in item.tag_set:
                return False
        if request.vegan and "vegan-option" not in item.tag_set:
            return False
        if request.gluten_free and "gluten" in item.tag_set:
            return False
        if request.halal and "halal" not in item.tag_set:
            return False
        if request.kosher and "kosher" not in item.tag_set:
            return False
        if request.dairy_free and "milk" in item.tag_set:
            return False
        if request.low_carbon and "low carbon footprint" not in icons_lower:
            return False

        if (request.vegetarian or request.vegan) and self._contains_meat_terms(item):
            return False

        if self._is_meal_request(normalized) and self._is_dessert_category(item.category):
            return False

        return True

    def _is_serving_now(self, item: MenuItem, now: dt.datetime, hint: Optional[str]) -> bool:
        local_now = now.astimezone(dt.timezone(dt.timedelta(hours=-7)))
        has_windows = False
        upcoming_window = False
        bucket_match = False
        for window in item.service_windows or []:
            start = window.get("start_time")
            end = window.get("end_time")
            if not (start and end):
                continue
            has_windows = True
            served_date = window.get("served_date") or window.get("served_date_raw")
            target_date = local_now.date()
            if served_date:
                try:
                    target_date = dt.datetime.strptime(served_date, "%Y-%m-%d").date()
                except ValueError:
                    pass
            start_dt = dt.datetime.combine(target_date, self._parse_time(start), local_now.tzinfo)
            end_dt = dt.datetime.combine(target_date, self._parse_time(end), local_now.tzinfo)
            if start_dt <= local_now <= end_dt:
                return True
            if start_dt > local_now:
                if (start_dt - local_now) <= dt.timedelta(hours=6):
                    upcoming_window = True
            window_bucket = self._bucketize_label(window.get("meal_period"))
            if hint and window_bucket == hint:
                bucket_match = True
        if has_windows:
            if bucket_match:
                return True
            if upcoming_window:
                return True
            return False

        if item.hours_structured:
            current_time = local_now.time()
            for block in item.hours_structured:
                start = block.get("start_time")
                end = block.get("end_time")
                if not (start and end):
                    continue
                start_time = self._parse_time(start)
                end_time = self._parse_time(end)
                if self._time_in_range(start_time, end_time, current_time, hint):
                    return True
            return False

        return True

    @staticmethod
    def _parse_time(value: str) -> dt.time:
        try:
            hour_str, minute_str = value.split(":", 1)
            return dt.time(int(hour_str), int(minute_str))
        except Exception:
            return dt.time(0, 0)

    def _time_in_range(self, start: dt.time, end: dt.time, current: dt.time, hint: Optional[str]) -> bool:
        if hint:
            desired = self._bucket_to_range(hint)
            if desired:
                return start == desired[0] and end == desired[1]
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _extract_locations(self, query: str) -> Set[str]:
        normalized = query.lower()
        return {canonical for key, canonical in KNOWN_LOCATIONS.items() if key in normalized}

    def _augment_query_terms(self, query: str) -> str:
        normalized = query.lower()
        additions: List[str] = []
        if "meaty" in normalized or "meat lover" in normalized or "meat lover's" in normalized:
            additions.extend(["meat", "beef", "chicken", "pork", "steak"])
        if "hearty" in normalized:
            additions.append("filling")
        if not additions:
            return query
        return f"{query} {' '.join(additions)}"

    def _extract_time_hints(self, query: str) -> Optional[str]:
        normalized = query.lower()
        if any(word in normalized for word in ["breakfast", "morning"]):
            return "breakfast"
        if "brunch" in normalized:
            return "lunch"
        if "lunch" in normalized:
            return "lunch"
        if any(word in normalized for word in ["dinner", "evening"]):
            return "dinner"
        if any(word in normalized for word in ["late-night", "late night", "midnight", "night"]):
            return "late-night"
        return None

    def _build_request_profile(self, normalized_query: str) -> RequestProfile:
        profile = RequestProfile(normalized_query=normalized_query)

        def contains(*phrases: str) -> bool:
            return any(phrase in normalized_query for phrase in phrases)

        profile.vegetarian = contains("vegetarian", "meatless", "veggie")
        profile.vegan = contains("vegan", "plant-based")
        profile.gluten_free = contains("gluten-free", "gluten free")
        profile.halal = "halal" in normalized_query
        profile.kosher = "kosher" in normalized_query
        profile.dairy_free = contains("dairy-free", "dairy free", "lactose free")
        profile.low_carbon = contains("low carbon", "low-carbon", "carbon footprint")
        profile.healthy_focus = self._has_health_focus(normalized_query)
        return profile

    @staticmethod
    def _bucketize_label(label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        text = label.lower()
        if "breakfast" in text or "morning" in text:
            return "breakfast"
        if "lunch" in text:
            return "lunch"
        if "late" in text or "midnight" in text:
            return "late-night"
        if "dinner" in text or "evening" in text:
            return "dinner"
        return None

    def _location_boost(self, normalized_query: str, location: str) -> float:
        boost = 0.0
        for key, canonical in KNOWN_LOCATIONS.items():
            if canonical == location and key in normalized_query:
                boost += 2.0
        return boost

    @staticmethod
    def _contains_meat_terms(item: MenuItem) -> bool:
        haystack = f"{item.name} {item.category}".lower()
        haystack += f" {item.doc_text.lower()}"
        return any(token in haystack for token in MEAT_KEYWORDS)

    @staticmethod
    def _is_meal_request(normalized: str) -> bool:
        return any(word in normalized for word in ["meal", "dinner", "lunch", "main", "entree", "plate"])

    @staticmethod
    def _is_dessert_category(category: str) -> bool:
        text = category.lower() if category else ""
        return any(word in text for word in DESSERT_CATEGORIES)

    @staticmethod
    def _bucket_to_range(bucket: str) -> Optional[Tuple[dt.time, dt.time]]:
        mapping = {
            "breakfast": (dt.time(7, 0), dt.time(10, 30)),
            "lunch": (dt.time(11, 0), dt.time(16, 0)),
            "dinner": (dt.time(16, 30), dt.time(21, 30)),
            "late-night": (dt.time(21, 30), dt.time(23, 59)),
        }
        return mapping.get(bucket)

    @staticmethod
    def _has_health_focus(text: str) -> bool:
        health_terms = [
            "healthy",
            "nutritious",
            "macro",
            "macros",
            "protein",
            "high protein",
            "gain",
            "post workout",
            "light",
            "low calorie",
            "balanced",
            "clean",
        ]
        return any(term in text for term in health_terms)

    def _nutrition_priority_boost(self, item: MenuItem, request: RequestProfile) -> float:
        """
        Reward entrees that deliver solid macros once the hard constraints
        (dietary/location) are satisfied. Desserts requested explicitly keep their rank.
        """
        normalized_query = request.normalized_query
        if self._is_dessert_category(item.category) and "dessert" in normalized_query:
            return 0.0

        emphasis = 0.35
        if request.healthy_focus:
            emphasis += 0.4
        if request.any_classification_requested:
            emphasis += 0.25

        for key, canonical in KNOWN_LOCATIONS.items():
            if key in normalized_query and canonical.lower() == item.location.lower():
                emphasis += 0.2
                break

        boost = 0.0
        protein = item.nutrition.get("Protein (g)", 0.0)
        fiber = item.nutrition.get("Total Dietary Fiber (g)", 0.0)
        calories = item.nutrition.get("Calories (kcal)")
        sodium = item.nutrition.get("Sodium (mg)")

        if protein:
            boost += min(protein / 8.0, 2.5)
            if protein < 6:
                boost -= 1.25
            elif protein >= 20:
                boost += 0.5
        else:
            boost -= 1.5
        if fiber:
            boost += min(fiber / 5.0, 1.0)
        elif item.category.lower().startswith("salad") is False:
            boost -= 0.25
        if calories:
            if 350 <= calories <= 750:
                boost += 0.5
            elif calories < 250:
                boost -= 0.25
            elif calories > 900:
                boost -= 0.25
        if sodium and sodium > 900:
            boost -= min((sodium - 900) / 400, 0.75)

        return boost * emphasis

    def _format_result(self, item: MenuItem, score: float, request: RequestProfile) -> Dict[str, Any]:
        nutrition_summary = self._build_nutrition_summary(item.nutrition)
        description = self._build_description(
            item,
            nutrition_summary,
            include_dietary=request.any_classification_requested,
        )
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
            "status": item.status,
            "crowdedness": item.crowdedness,
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
    def _build_description(
        item: MenuItem,
        nutrition_summary: Dict[str, float],
        include_dietary: bool,
    ) -> str:
        pieces = [
            f"{item.name} at {item.location} ({item.meal})",
        ]
        if item.category:
            pieces.append(f"Category: {item.category}.")

        if include_dietary and item.dietary_choices:
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
