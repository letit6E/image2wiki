"""
Асинхронный сбор данных с ru.wikipedia.org для fine-tuning.
Ускоряет исходный [collect_data.py](collect_data.py) за счет конкурентной загрузки картинок,
но оставляет API-запросы к Wikipedia достаточно бережными.

Установка:
    pip install aiohttp tqdm

Примеры:
    python collect_data_async.py
    python collect_data_async.py --max-total 10000 --max-depth 2 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, AsyncIterator, TextIO
from urllib.parse import unquote

import aiohttp
from tqdm import tqdm

API_URL = "https://ru.wikipedia.org/w/api.php"
HEADERS = {
    # Укажи свои контакты при желании; для Wikimedia лучше честный bot UA, а не браузерный.
    "User-Agent": "ML2HomeworkCollector/1.0 (educational project; contact: local-run)",
    "Accept-Encoding": "gzip, deflate",
}
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)

CATEGORIES = [
    "Категория:Кошки (род)",
    "Категория:Породы собак",
    "Категория:Совообразные",
    "Категория:Попугаеобразные",
    "Категория:Ястребообразные",
    "Категория:Бабочки",
    "Категория:Жуки",
    "Категория:Пресноводные рыбы",
    "Категория:Акулы",
    "Категория:Съедобные грибы",
    "Категория:Ядовитые грибы",
    "Категория:Фрукты",
    "Категория:Овощи",
    "Категория:Ягоды",
    "Категория:Орехи",
    "Категория:Хвойные",
    "Категория:Цветковые растения",
    "Категория:Кактусовые",
    "Категория:Легковые автомобили",
    "Категория:Мотоциклы",
    "Категория:Вертолёты",
    "Категория:Самолёты",
    "Категория:Танки",
    "Категория:Струнные музыкальные инструменты",
    "Категория:Духовые музыкальные инструменты",
    "Категория:Ударные музыкальные инструменты",
    "Категория:Супы",
    "Категория:Салаты",
    "Категория:Пирожные",
    "Категория:Мосты России",
    "Категория:Мосты Европы",
    "Категория:Замки Европы",
    "Категория:Храмы России",
    "Категория:Небоскрёбы",
    "Категория:Маяки",
    "Категория:Вулканы",
    "Категория:Озёра России",
    "Категория:Водопады",
    "Категория:Холодное оружие",
    "Категория:Огнестрельное оружие",
    "Категория:Минералы",
    "Категория:Драгоценные камни",
    "Категория:Монеты",
    "Категория:Флаги государств",
]

SKIP_IMAGE_EXTENSIONS = {".svg", ".gif", ".ogg", ".ogv", ".webm", ".pdf", ".djvu"}

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.jsonl"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"


class AsyncCollector:
    def __init__(self, max_total: int, max_depth: int, resume: bool):
        self.max_total = max_total
        self.max_depth = max_depth
        self.resume = resume
        self.collected: set[str] = set()
        self.session: aiohttp.ClientSession | None = None
        self.meta_f: TextIO | None = None
        self.pbar: tqdm | None = None

        # API лучше не долбить параллельно; ускорение в основном будет на картинках.
        self.api_sem = asyncio.Semaphore(1)
        self.img_sem = asyncio.Semaphore(8)

    async def init(self) -> None:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        connector = aiohttp.TCPConnector(limit=16)
        self.session = aiohttp.ClientSession(
            headers=HEADERS,
            connector=connector,
            timeout=REQUEST_TIMEOUT,
        )

        if self.resume and CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                self.collected = set(json.load(f).get("collected_titles", []))

        mode = "a" if self.resume and METADATA_FILE.exists() else "w"
        self.meta_f = open(METADATA_FILE, mode, encoding="utf-8")
        self.pbar = tqdm(total=self.max_total, initial=len(self.collected), desc="Collecting")

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
        if self.meta_f is not None:
            self.meta_f.close()
        if self.pbar is not None:
            self.pbar.close()

    def save_checkpoint(self) -> None:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"collected_titles": list(self.collected)}, f, ensure_ascii=False)

    async def api_query(self, **params: Any) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("Session is not initialized")

        normalized_params: dict[str, str | int | float] = {
            "format": "json",
            "action": "query",
        }
        for key, value in params.items():
            if isinstance(value, bool):
                normalized_params[key] = "1" if value else "0"
            elif isinstance(value, (str, int, float)):
                normalized_params[key] = value
            else:
                normalized_params[key] = str(value)

        async with self.api_sem:
            await asyncio.sleep(0.05)
            for attempt in range(4):
                try:
                    async with self.session.get(API_URL, params=normalized_params) as resp:
                        if resp.status in (403, 429):
                            wait = int(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                            tqdm.write(f"API limited ({resp.status}), sleeping {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:
                    if attempt == 3:
                        tqdm.write(f"API Error: {e}")
                        return {}
                    await asyncio.sleep(1.5 * (attempt + 1))
        return {}

    async def download_image(self, url: str, save_path: Path) -> bool:
        if self.session is None:
            raise RuntimeError("Session is not initialized")
        if save_path.exists():
            return True

        async with self.img_sem:
            for attempt in range(3):
                try:
                    async with self.session.get(url) as resp:
                        if resp.status in (403, 429):
                            wait = int(resp.headers.get("Retry-After", 3 * (attempt + 1)))
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        content = await resp.read()
                        with open(save_path, "wb") as f:
                            f.write(content)
                        return True
                except Exception:
                    if attempt == 2:
                        return False
                    await asyncio.sleep(1 + attempt)
        return False

    async def iter_category_pages(self, category: str, max_per_category: int) -> AsyncIterator[str]:
        visited_cats: set[str] = set()
        count = 0

        async def _crawl(cat: str, depth: int) -> AsyncIterator[str]:
            nonlocal count
            if depth > self.max_depth or cat in visited_cats or count >= max_per_category:
                return
            visited_cats.add(cat)

            cmcontinue: str | None = None
            subcats: list[str] = []

            while count < max_per_category:
                params: dict[str, Any] = {
                    "list": "categorymembers",
                    "cmtitle": cat,
                    "cmlimit": 50,
                    "cmtype": "page|subcat",
                }
                if cmcontinue:
                    params["cmcontinue"] = cmcontinue

                data = await self.api_query(**params)
                members = data.get("query", {}).get("categorymembers", [])
                if not members and "error" in data:
                    return

                for member in members:
                    if count >= max_per_category:
                        return
                    if member.get("ns") == 0:
                        title = member.get("title")
                        if isinstance(title, str):
                            count += 1
                            yield title
                    elif member.get("ns") == 14:
                        title = member.get("title")
                        if isinstance(title, str):
                            subcats.append(title)

                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue:
                    break

            for subcat in subcats:
                if count >= max_per_category:
                    return
                async for title in _crawl(subcat, depth + 1):
                    yield title

        async for title in _crawl(category, 0):
            yield title

    async def process_batch(self, batch: list[str], category: str) -> int:
        data = await self.api_query(
            titles="|".join(batch),
            prop="extracts|pageimages",
            exintro=1,
            explaintext=1,
            exsectionformat="plain",
            piprop="thumbnail",
            pithumbsize=512,
            pilimit="max",
        )
        pages = data.get("query", {}).get("pages", {})

        tasks: list[asyncio.Task[bool]] = []
        records: list[dict[str, str]] = []

        for page_id, page in pages.items():
            try:
                if int(page_id) < 0:
                    continue
            except Exception:
                continue

            title = page.get("title", "")
            if not isinstance(title, str) or title in self.collected:
                continue

            extract = page.get("extract", "")
            thumb = page.get("thumbnail", {})
            image_url = thumb.get("source", "") if isinstance(thumb, dict) else ""

            if not isinstance(extract, str) or len(extract.strip()) < 50:
                continue
            if not isinstance(image_url, str) or not image_url:
                continue

            ext = Path(unquote(image_url)).suffix.lower().split("?")[0]
            if ext in SKIP_IMAGE_EXTENSIONS:
                continue

            safe_name = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
            final_ext = ext if ext and len(ext) <= 5 else ".jpg"
            img_path = IMAGES_DIR / f"{safe_name}{final_ext}"

            records.append(
                {
                    "title": title,
                    "text": extract.strip(),
                    "image_path": str(img_path),
                    "image_url": image_url,
                    "category": category,
                }
            )
            tasks.append(asyncio.create_task(self.download_image(image_url, img_path)))

        if not tasks:
            return 0

        results = await asyncio.gather(*tasks)

        if self.meta_f is None or self.pbar is None:
            raise RuntimeError("Output files are not initialized")

        added = 0
        for record, success in zip(records, results):
            if not success:
                continue
            self.meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.collected.add(record["title"])
            added += 1
            self.pbar.update(1)

        self.meta_f.flush()
        return added

    async def collect_from_category(self, category: str, limit: int) -> int:
        cat_count = 0
        batch: list[str] = []

        async for title in self.iter_category_pages(category, limit * 3):
            if cat_count >= limit or len(self.collected) >= self.max_total:
                break
            if title in self.collected:
                continue

            batch.append(title)
            if len(batch) >= 50:
                cat_count += await self.process_batch(batch, category)
                batch = []

        if batch and cat_count < limit and len(self.collected) < self.max_total:
            cat_count += await self.process_batch(batch, category)

        return cat_count

    async def run(self) -> None:
        await self.init()
        try:
            base_per_cat = self.max_total // len(CATEGORIES)
            tqdm.write(f"Pass 1: up to {base_per_cat} per category ({len(CATEGORIES)} categories)")

            cat_stats: dict[str, int] = {}
            for category in CATEGORIES:
                if len(self.collected) >= self.max_total:
                    break
                tqdm.write(f"\n📂 {category}")
                n = await self.collect_from_category(category, base_per_cat)
                cat_stats[category] = n
                tqdm.write(f"   ✓ {n} pairs")
                self.save_checkpoint()

            remaining = self.max_total - len(self.collected)
            if remaining > 0:
                big_cats = sorted(cat_stats, key=lambda c: cat_stats[c], reverse=True)
                extra_per_cat = remaining // min(len(big_cats), 10) + 50

                tqdm.write(f"\nPass 2: collecting {remaining} more from largest categories")
                for category in big_cats:
                    if len(self.collected) >= self.max_total:
                        break
                    tqdm.write(f"\n📂 {category} (extra)")
                    n = await self.collect_from_category(category, extra_per_cat)
                    tqdm.write(f"   ✓ {n} extra pairs")
                    self.save_checkpoint()
        finally:
            await self.close()

        print(f"\nDone! Collected {len(self.collected)} pairs.")
        print(f"Images: {IMAGES_DIR}")
        print(f"Metadata: {METADATA_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Wikipedia image-text pairs (async)")
    parser.add_argument("--max-total", type=int, default=10000, help="Total pairs to collect")
    parser.add_argument("--max-depth", type=int, default=2, help="Max category recursion depth")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    collector = AsyncCollector(args.max_total, args.max_depth, args.resume)
    asyncio.run(collector.run())


if __name__ == "__main__":
    main()
