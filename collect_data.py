"""
Сбор данных с ru.wikipedia.org для fine-tuning CLIP.
Собирает случайные статьи с изображениями — пары (картинка, текст).

Использование:
    python collect_data.py
    python collect_data.py --max-total 10000
    python collect_data.py --max-total 10000 --resume
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from tqdm import tqdm

API_URL = "https://ru.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://ru.wikipedia.org/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
})

SKIP_IMAGE_EXTENSIONS = {".svg", ".gif", ".ogg", ".ogv", ".webm", ".pdf", ".djvu"}

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.jsonl"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"


def api_query(**params):
    """Запрос к MediaWiki API с rate limiting."""
    params.setdefault("format", "json")
    params.setdefault("action", "query")
    time.sleep(0.1)
    resp = SESSION.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_random_titles(count: int = 20) -> list[str]:
    """Получить случайные заголовки статей (namespace 0 = основные статьи)."""
    data = api_query(list="random", rnnamespace=0, rnlimit=count)
    return [p["title"] for p in data.get("query", {}).get("random", [])]


def get_article_data(titles: list[str]) -> dict:
    """Получить extract + thumbnail для пачки статей (до 20)."""
    data = api_query(
        titles="|".join(titles),
        prop="extracts|pageimages",
        exintro=True,
        explaintext=True,
        exsectionformat="plain",
        piprop="thumbnail",
        pithumbsize=512,
        pilimit="max",
    )
    pages = data.get("query", {}).get("pages", {})
    results = {}
    for page_id, page in pages.items():
        if int(page_id) < 0:
            continue
        title = page.get("title", "")
        extract = page.get("extract", "").strip()
        thumb = page.get("thumbnail", {})
        image_url = thumb.get("source", "")
        results[title] = {"extract": extract, "image_url": image_url}
    return results


def download_image(url: str, save_path: Path, max_retries: int = 3) -> bool:
    """Скачать изображение с retry и exponential backoff."""
    for attempt in range(max_retries):
        try:
            time.sleep(0.2 + attempt * 2)
            resp = SESSION.get(url, timeout=30, stream=True)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                tqdm.write(f"  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            tqdm.write(f"  ⚠ Download failed: {e}")
            return False
        except Exception as e:
            tqdm.write(f"  ⚠ Download failed: {e}")
            return False
    return False


def image_filename(title: str, url: str) -> str:
    ext = Path(unquote(url)).suffix.lower().split("?")[0]
    if not ext or len(ext) > 5:
        ext = ".jpg"
    safe_name = hashlib.md5(title.encode()).hexdigest()[:12]
    return f"{safe_name}{ext}"


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f).get("collected_titles", []))
    return set()


def save_checkpoint(collected: set[str]):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"collected_titles": list(collected)}, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Collect random Wikipedia image-text pairs")
    parser.add_argument("--max-total", type=int, default=10000, help="Total pairs to collect")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    collected = load_checkpoint() if args.resume else set()
    mode = "a" if args.resume and METADATA_FILE.exists() else "w"

    total = len(collected)
    skipped = 0
    pbar = tqdm(total=args.max_total, initial=total, desc="Collecting")

    with open(METADATA_FILE, mode, encoding="utf-8") as meta_f:
        while total < args.max_total:
            # Берём пачку случайных статей
            random_titles = get_random_titles(20)
            # Фильтруем уже собранные
            new_titles = [t for t in random_titles if t not in collected]
            if not new_titles:
                continue

            # Получаем данные статей
            article_data = get_article_data(new_titles)

            for title, info in article_data.items():
                if total >= args.max_total:
                    break
                if title in collected:
                    continue

                extract = info["extract"]
                image_url = info["image_url"]

                # Пропуск статей без текста или картинки
                if not extract or len(extract) < 50:
                    skipped += 1
                    continue
                if not image_url:
                    skipped += 1
                    continue

                # Пропуск не-фото форматов
                ext = Path(unquote(image_url)).suffix.lower().split("?")[0]
                if ext in SKIP_IMAGE_EXTENSIONS:
                    skipped += 1
                    continue

                # Скачиваем
                fname = image_filename(title, image_url)
                img_path = IMAGES_DIR / fname

                if not img_path.exists():
                    if not download_image(image_url, img_path):
                        skipped += 1
                        continue

                record = {
                    "title": title,
                    "text": extract,
                    "image_path": str(img_path),
                    "image_url": image_url,
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()

                collected.add(title)
                total += 1
                pbar.update(1)

            # Checkpoint каждые 100 статей
            if total % 100 < 20:
                save_checkpoint(collected)
                pbar.set_postfix(skipped=skipped)

    save_checkpoint(collected)
    pbar.close()
    print(f"\nDone! Collected {total} pairs (skipped {skipped} without image/text).")
    print(f"Images: {IMAGES_DIR}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
