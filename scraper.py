#!/usr/bin/env python3
"""Scrape UC Berkeley dining hall menus into a structured JSON file."""

from __future__ import annotations

import argparse
import base64
import time as time_module
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


MENU_URL = "https://dining.berkeley.edu/menus/"
DEFAULT_OUTPUT = Path("menus.json")


def fetch_html(url: str) -> str:
    """Retrieve the HTML for the menus page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def decode_menu_source(encoded: Optional[str]) -> Optional[str]:
    """Decode the base64-encoded XML source link for a menu item."""
    if not encoded:
        return None

    padded = encoded.strip()
    # The data-location attribute is base64 without padding; add it back if missing.
    padding = len(padded) % 4
    if padding:
        padded += "=" * (4 - padding)

    try:
        return base64.b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def text_or_none(element: Optional[Tag]) -> Optional[str]:
    """Return stripped text for a tag or None when missing."""
    if not element:
        return None
    value = element.get_text(strip=True)
    return value or None


def extract_meal_label(meal_el: Tag) -> Optional[str]:
    """Pull the visible label for a meal period."""
    label_span = meal_el.find("span")
    if not label_span:
        return None

    pieces: List[str] = []
    for child in label_span.contents:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                pieces.append(text)
    return " ".join(pieces) or None


def parse_icon(icon_el: Tag) -> Dict[str, Optional[str]]:
    """Extract metadata for a single informational icon."""
    img = icon_el.find("img")
    tooltip = icon_el.select_one(".allg-tooltip")

    label = text_or_none(tooltip)
    alt = img.get("alt") if img and img.has_attr("alt") else None
    src = img.get("src") if img and img.has_attr("src") else None

    return {
        "label": label or alt,
        "alt": alt,
        "image": src,
    }


def parse_icons(wrapper: Optional[Tag]) -> List[Dict[str, Optional[str]]]:
    """Collect all icon descriptors for a menu item."""
    if not wrapper:
        return []
    return [parse_icon(icon_el) for icon_el in wrapper.select(".food-icon")]


def parse_menu_item(item_el: Tag) -> Dict[str, Any]:
    """Convert an individual menu item entry into a dictionary."""
    # Name is the first direct child span without a class attribute.
    name_span = None
    for child in item_el.find_all("span", recursive=False):
        if not child.has_attr("class"):
            name_span = child
            break

    name = text_or_none(name_span) or item_el.get_text(" ", strip=True)

    icons = parse_icons(item_el.select_one(".icons-wrap"))

    classes = [cls for cls in item_el.get("class", []) if cls != "recip"]

    return {
        "name": name,
        "id": item_el.get("data-id"),
        "menu_id": item_el.get("data-menuid"),
        "menu_source": decode_menu_source(item_el.get("data-location")),
        "tags": classes,
        "icons": icons,
    }


def parse_category(cat_el: Tag) -> Dict[str, Any]:
    """Extract a category block inside a meal period."""
    category_name = text_or_none(cat_el.find("span"))
    items = [parse_menu_item(item) for item in cat_el.select("ul.recipe-name > li.recip")]
    return {
        "category": category_name,
        "items": items,
    }


def parse_meal_period(meal_el: Tag) -> Dict[str, Any]:
    """Extract the details for a single meal period."""
    label = extract_meal_label(meal_el)
    categories = [
        parse_category(cat)
        for cat in meal_el.select(".recipes-main-wrap > div.cat-name")
    ]
    return {
        "label": label,
        "categories": categories,
    }


def parse_location(location_el: Tag) -> Dict[str, Any]:
    """Convert a dining location node into a dictionary."""
    name = text_or_none(location_el.select_one(".cafe-title"))
    status = text_or_none(location_el.select_one(".status"))
    serve_date = text_or_none(location_el.select_one(".serve-date"))
    hours = [
        span.get_text(strip=True)
        for span in location_el.select(".times span")
        if span.get_text(strip=True)
    ]

    classes = [cls for cls in location_el.get("class", []) if cls != "location-name"]
    slug = "-".join(classes) if classes else None

    meals = [
        parse_meal_period(meal_el)
        for meal_el in location_el.select("ul.meal-period > li")
    ]

    return {
        "name": name,
        "status": status,
        "serve_date": serve_date,
        "hours": hours,
        "slug": slug,
        "meals": meals,
    }


def parse_document(html: str) -> Dict[str, Any]:
    """Parse the complete menus page into structured data."""
    soup = BeautifulSoup(html, "html.parser")
    locations = [
        parse_location(location_el)
        for location_el in soup.select("ul.cafe-location > li.location-name")
    ]
    return {
        "source_url": MENU_URL,
        "locations": locations,
    }


def run_scrape(url: str, output_path: Path) -> Dict[str, Any]:
    """Perform a single scrape cycle and persist the JSON."""
    html = fetch_html(url)
    parsed = parse_document(html)
    output_path.write_text(json.dumps(parsed, indent=2))
    return parsed


def parse_daily_time(value: str) -> time:
    """Parse a HH:MM string into a time object."""
    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Time must be in HH:MM format") from None

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError("Hours must be 0-23 and minutes 0-59")

    return time(hour=hour, minute=minute)


def seconds_until(target: time, tz: ZoneInfo) -> float:
    """Compute seconds until the next occurrence of target time in the given timezone."""
    now = datetime.now(tz)
    today_target = datetime.combine(now.date(), target, tzinfo=tz)
    if today_target <= now:
        today_target += timedelta(days=1)
    return (today_target - now).total_seconds()


def run_daily(url: str, output_path: Path, target_time: time, tz_name: str) -> None:
    """Continuously run the scraper once per day at the requested time."""
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise SystemExit(f"Unknown timezone '{tz_name}'.") from exc

    while True:
        wait_seconds = seconds_until(target_time, tz)
        next_run = datetime.now(tz) + timedelta(seconds=wait_seconds)
        print(f"Next scrape scheduled for {next_run.isoformat(timespec='minutes')}")
        time_module.sleep(wait_seconds)
        try:
            data = run_scrape(url, output_path)
            print(
                f"Updated menu data for {len(data['locations'])} locations at "
                f"{datetime.now(tz).isoformat(timespec='minutes')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Scrape failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape the UC Berkeley dining menus page into JSON."
    )
    parser.add_argument(
        "--url",
        default=MENU_URL,
        help="Menus page URL (defaults to the Berkeley Dining menus page).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Run continuously and refresh the JSON once per day.",
    )
    parser.add_argument(
        "--time",
        type=parse_daily_time,
        default=parse_daily_time("00:01"),
        help="Target time for the daily refresh in HH:MM (24h) format (default: 00:01).",
    )
    parser.add_argument(
        "--timezone",
        default="America/Los_Angeles",
        help="Timezone name for scheduling daily refreshes (default: America/Los_Angeles).",
    )
    args = parser.parse_args()

    if args.daily:
        print(
            f"Starting daily scraper targeting {args.time.strftime('%H:%M')} "
            f"{args.timezone} and writing to {args.output}"
        )
        try:
            run_daily(args.url, args.output, args.time, args.timezone)
        except KeyboardInterrupt:
            print("Daily scraper stopped.")
    else:
        parsed = run_scrape(args.url, args.output)
        print(f"Saved menu data for {len(parsed['locations'])} locations to {args.output}")


if __name__ == "__main__":
    main()
