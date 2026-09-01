#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taiwan Streetscape Finder V2.2
==============================
Balanced, scalable metadata-first corpus harvester for Taiwanese streetscape research.

Main additions over V2.1:
- round-robin dense geographic grid across Taiwanese cities
- per-city acceptance cap to reduce geographic domination
- optional seed CSV import so an earlier candidate pool can be reused
- geo-grid parameters designed for 500–1000 final-scene corpus expansion

Retained from V2.1:
- Wikimedia 2026 rate-limit aware (Retry-After + exponential backoff)
- compliant User-Agent can be supplied with --wikimedia-contact
- safe request pacing (slower if no real contact is supplied)
- fewer Wikimedia requests by combining generators + imageinfo in one call
- geo/category search runs before keyword search
- Openverse 401 is handled gracefully; optional OAuth client credentials supported
- Openverse license_type filter is OFF by default; licenses remain recorded
- discovery checkpoint + --resume
- provenance-rich candidate manifests and diagnostics

This program discovers candidate images. It does not declare them training-ready.
Always verify license/provenance before publication or redistribution.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
OPENVERSE_TOKEN_API = "https://api.openverse.org/v1/auth_tokens/token/"
VERSION = "2.2"

DEFAULT_KEYWORD_QUERIES = [
    "Taiwan street", "Taiwan streetscape", "Taiwan urban street", "Taiwan alley",
    "Taiwan residential street", "Taiwan commercial street", "Taiwan shophouse",
    "Taiwan apartment facade", "Taiwan arcade street", "Taiwan storefront",
    "台灣 街景", "台灣 街道", "台灣 巷弄", "台灣 騎樓", "台灣 店面", "台灣 公寓 立面",
    "Taipei street", "Taipei alley", "Taipei residential street", "New Taipei street",
    "Keelung street", "Taoyuan street", "Hsinchu street", "Taichung street",
    "Changhua street", "Chiayi street", "Tainan street", "Kaohsiung street",
    "Pingtung street", "Yilan street", "Hualien street", "Taitung street",
    "台北 街道", "新北 街道", "基隆 街道", "桃園 街道", "新竹 街道", "台中 街道",
    "彰化 街道", "嘉義 街道", "台南 街道", "高雄 街道", "屏東 街道", "宜蘭 街道",
    "花蓮 街道", "台東 街道",
]

DEFAULT_CATEGORIES = [
    "Category:Streets in Taiwan",
    "Category:Roads in Taiwan",
    "Category:Streets in Taipei",
    "Category:Roads in Taipei",
    "Category:Streets in New Taipei",
    "Category:Roads in New Taipei",
    "Category:Streets in Taichung",
    "Category:Roads in Taichung",
    "Category:Streets in Tainan",
    "Category:Roads in Tainan",
    "Category:Streets in Kaohsiung",
    "Category:Roads in Kaohsiung",
    "Category:Alleys in Taiwan",
    "Category:Buildings and structures in Taiwan by street",
]

DEFAULT_GEO_CENTERS = [
    {"name": "Taipei", "lat": 25.0330, "lon": 121.5654},
    {"name": "NewTaipei", "lat": 25.0169, "lon": 121.4628},
    {"name": "Keelung", "lat": 25.1276, "lon": 121.7392},
    {"name": "Taoyuan", "lat": 24.9937, "lon": 121.3010},
    {"name": "Hsinchu", "lat": 24.8138, "lon": 120.9675},
    {"name": "Miaoli", "lat": 24.5602, "lon": 120.8214},
    {"name": "Taichung", "lat": 24.1477, "lon": 120.6736},
    {"name": "Changhua", "lat": 24.0817, "lon": 120.5380},
    {"name": "Nantou", "lat": 23.9609, "lon": 120.9719},
    {"name": "Yunlin", "lat": 23.7092, "lon": 120.5430},
    {"name": "Chiayi", "lat": 23.4801, "lon": 120.4491},
    {"name": "Tainan", "lat": 22.9999, "lon": 120.2270},
    {"name": "Kaohsiung", "lat": 22.6273, "lon": 120.3014},
    {"name": "Pingtung", "lat": 22.5519, "lon": 120.5488},
    {"name": "Yilan", "lat": 24.7570, "lon": 121.7530},
    {"name": "Hualien", "lat": 23.9872, "lon": 121.6015},
    {"name": "Taitung", "lat": 22.7583, "lon": 121.1444},
]

# City-specific grid spacing. The aim is not geographic surveying precision; it is
# to sample distinct urban neighborhoods without letting one city dominate.
MAJOR_GRID_STEP_KM = {
    "Taipei": 3.2, "NewTaipei": 4.0, "Taoyuan": 4.0, "Taichung": 4.5,
    "Tainan": 4.0, "Kaohsiung": 4.5,
}
MEDIUM_GRID_STEP_KM = {
    "Keelung": 2.5, "Hsinchu": 3.0, "Miaoli": 2.5, "Changhua": 2.8,
    "Nantou": 2.5, "Yunlin": 2.8, "Chiayi": 2.8, "Pingtung": 3.0,
    "Yilan": 2.8, "Hualien": 2.8, "Taitung": 2.8,
}

# Ordered from central to peripheral. generate_geo_grid() applies each offset to
# *all* cities before moving to the next offset, producing a round-robin corpus.
GEO_GRID_OFFSETS = [
    ("C", 0.0, 0.0),
    ("N", 0.0, 1.0), ("E", 1.0, 0.0), ("S", 0.0, -1.0), ("W", -1.0, 0.0),
    ("NE", 1.0, 1.0), ("SE", 1.0, -1.0), ("SW", -1.0, -1.0), ("NW", -1.0, 1.0),
    ("N2", 0.0, 2.0), ("E2", 2.0, 0.0), ("S2", 0.0, -2.0), ("W2", -2.0, 0.0),
]


NEGATIVE_TEXT_TERMS = {
    "map", "diagram", "drawing", "illustration", "painting", "poster", "logo",
    "flag", "coat of arms", "satellite", "aerial", "drone", "floor plan",
    "interior", "inside", "museum exhibit", "portrait", "food", "dish",
    "ticket", "document", "screenshot", "scan",
    "地圖", "示意圖", "平面圖", "插畫", "繪畫", "海報", "室內", "空拍",
}

LICENSE_FIELDS = [
    "LicenseShortName", "LicenseUrl", "UsageTerms", "Artist", "Credit",
    "ImageDescription", "Attribution",
]

POSITIVE_CLIP_TEXTS = [
    "a street-level photograph of an urban street in Taiwan",
    "a Taiwanese streetscape with buildings, storefronts, signs and sidewalks",
    "a street-level view of residential buildings in Taiwan",
    "a Taiwanese alley with building facades",
    "a street scene in a Taiwanese city",
]
NEGATIVE_CLIP_TEXTS = [
    "an indoor room", "an aerial or satellite photograph",
    "a landscape without an urban street", "a map or diagram",
    "a poster, illustration or artwork", "a close-up photograph of a single object",
    "a portrait photograph", "a food photograph",
]

CITY_HINTS = {
    "taipei": "Taipei", "台北": "Taipei", "new taipei": "NewTaipei", "新北": "NewTaipei",
    "keelung": "Keelung", "基隆": "Keelung", "taoyuan": "Taoyuan", "桃園": "Taoyuan",
    "hsinchu": "Hsinchu", "新竹": "Hsinchu", "miaoli": "Miaoli", "苗栗": "Miaoli",
    "taichung": "Taichung", "台中": "Taichung", "changhua": "Changhua", "彰化": "Changhua",
    "nantou": "Nantou", "南投": "Nantou", "yunlin": "Yunlin", "雲林": "Yunlin",
    "chiayi": "Chiayi", "嘉義": "Chiayi", "tainan": "Tainan", "台南": "Tainan",
    "kaohsiung": "Kaohsiung", "高雄": "Kaohsiung", "pingtung": "Pingtung", "屏東": "Pingtung",
    "yilan": "Yilan", "宜蘭": "Yilan", "hualien": "Hualien", "花蓮": "Hualien",
    "taitung": "Taitung", "台東": "Taitung", "taiwan": "Taiwan", "台灣": "Taiwan",
}


def clean_html(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def guess_city(text: str) -> str:
    t = (text or "").lower()
    for key, value in CITY_HINTS.items():
        if key in t:
            return value
    return ""


def slugify(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "item"


@dataclass
class Candidate:
    provider: str
    provider_id: str
    query: str
    search_mode: str
    title: str
    source_page: str
    image_url: str
    thumbnail_url: str
    width: int
    height: int
    license: str
    license_url: str
    creator: str
    attribution: str
    description: str
    source_name: str = ""
    category_path: str = ""
    geo_center_name: str = ""
    geo_lat: Optional[float] = None
    geo_lon: Optional[float] = None
    guessed_city: str = ""
    downloaded_path: str = ""
    sha256: str = ""
    dhash: str = ""
    clip_street_probability: Optional[float] = None
    status: str = "candidate"
    rejection_reason: str = ""
    extra: str = ""


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = max(0.0, float(delay))
        self.last = 0.0

    def wait(self):
        elapsed = time.time() - self.last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last = time.time()


class Finder:
    def __init__(self, args, out_dir: Path):
        self.args = args
        self.out_dir = out_dir
        self.session = requests.Session()
        self.session.headers.update({"Accept-Encoding": "gzip"})

        contact = (args.wikimedia_contact or os.environ.get("WIKIMEDIA_CONTACT", "")).strip()
        if contact:
            ua = f"PlayingModelsTaiwanStreetscapeFinder/{VERSION} (academic research; {contact})"
            wikimedia_delay = max(args.wikimedia_request_delay, 1.0)
            print(f"Wikimedia contact: configured | request delay={wikimedia_delay:.2f}s")
        else:
            ua = f"PlayingModelsTaiwanStreetscapeFinder/{VERSION} (academic research)"
            wikimedia_delay = max(args.wikimedia_request_delay, 6.5)
            print("WARNING: --wikimedia-contact not supplied.")
            print("Using conservative Wikimedia pacing (~9 requests/min).")
            print("For faster compliant access, pass a real email or URL locally via --wikimedia-contact.")
        self.session.headers.update({"User-Agent": ua})
        self.wikimedia_rate = RateLimiter(wikimedia_delay)
        self.openverse_rate = RateLimiter(max(args.openverse_request_delay, 1.5))

        self.openverse_token = os.environ.get("OPENVERSE_ACCESS_TOKEN", "").strip()
        self.openverse_disabled = False
        self.openverse_auth_attempted = False

        self.seen_provider_keys: Set[Tuple[str, str]] = set()
        self.seen_image_urls: Set[str] = set()
        self.accepted: List[Candidate] = []
        self.rejected: List[Candidate] = []
        self.completed_units: Set[str] = set()

        self.provider_stats = Counter()
        self.mode_stats = Counter()
        self.query_stats = Counter()
        self.reject_reason_stats = Counter()
        self.accepted_city_counts = Counter()
        self.seed_count = 0
        self.http_status_stats = Counter()
        self.retry_events = []
        self.openverse_debug = []

        if args.resume:
            self.load_checkpoint()

    @property
    def checkpoint_path(self) -> Path:
        return self.out_dir / "discovery_checkpoint.json"

    def save_checkpoint(self):
        payload = {
            "version": VERSION,
            "accepted": [asdict(x) for x in self.accepted],
            "rejected": [asdict(x) for x in self.rejected],
            "completed_units": sorted(self.completed_units),
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            print("--resume requested, but no checkpoint exists; starting fresh.")
            return
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        self.accepted = [Candidate(**x) for x in payload.get("accepted", [])]
        self.rejected = [Candidate(**x) for x in payload.get("rejected", [])]
        self.completed_units = set(payload.get("completed_units", []))
        for c in self.accepted + self.rejected:
            if c.provider_id:
                self.seen_provider_keys.add((c.provider, c.provider_id))
            if c.image_url:
                self.seen_image_urls.add(c.image_url)
        self.accepted_city_counts = Counter((c.guessed_city or "Unknown") for c in self.accepted)
        print(f"Resumed checkpoint: accepted={len(self.accepted)} rejected={len(self.rejected)} completed_units={len(self.completed_units)}")

    def _parse_retry_after(self, response: requests.Response, attempt: int) -> float:
        raw = response.headers.get("Retry-After", "").strip()
        if raw:
            try:
                return max(1.0, min(float(raw), self.args.max_retry_wait))
            except ValueError:
                pass
        return min(self.args.backoff_base * (2 ** attempt), self.args.max_retry_wait)

    def _request(self, url: str, params: Optional[dict], service: str,
                 method: str = "GET", data: Optional[dict] = None,
                 timeout: int = 45) -> requests.Response:
        limiter = self.wikimedia_rate if service == "wikimedia" else self.openverse_rate
        headers = {}
        if service == "openverse" and self.openverse_token:
            headers["Authorization"] = f"Bearer {self.openverse_token}"

        last_exc = None
        for attempt in range(self.args.max_retries + 1):
            limiter.wait()
            try:
                if method == "POST":
                    resp = self.session.post(url, params=params, data=data, headers=headers, timeout=timeout)
                else:
                    resp = self.session.get(url, params=params, headers=headers, timeout=timeout)
                self.http_status_stats[f"{service}:{resp.status_code}"] += 1

                if resp.status_code in (429, 503):
                    wait = self._parse_retry_after(resp, attempt)
                    event = {"service": service, "status": resp.status_code, "wait_seconds": wait, "attempt": attempt + 1}
                    self.retry_events.append(event)
                    print(f"[{service}] HTTP {resp.status_code}; waiting {wait:.1f}s before retry {attempt+1}/{self.args.max_retries}")
                    time.sleep(wait)
                    continue

                # Openverse anonymous request unexpectedly returning 401: optionally authenticate.
                if service == "openverse" and resp.status_code == 401 and not self.openverse_token:
                    if self.try_openverse_authentication():
                        headers["Authorization"] = f"Bearer {self.openverse_token}"
                        continue
                    self.openverse_disabled = True
                    raise RuntimeError(
                        "Openverse returned HTTP 401 and no usable credentials were available. "
                        "Openverse is disabled for this run; Wikimedia search will continue."
                    )

                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.args.max_retries:
                    break
                wait = min(self.args.backoff_base * (2 ** attempt), self.args.max_retry_wait)
                print(f"[{service}] request error {type(exc).__name__}; waiting {wait:.1f}s")
                time.sleep(wait)
        if last_exc:
            raise last_exc
        raise RuntimeError(f"{service} request failed after retries")

    def try_openverse_authentication(self) -> bool:
        if self.openverse_auth_attempted:
            return False
        self.openverse_auth_attempted = True
        client_id = (self.args.openverse_client_id or os.environ.get("OPENVERSE_CLIENT_ID", "")).strip()
        client_secret = (self.args.openverse_client_secret or os.environ.get("OPENVERSE_CLIENT_SECRET", "")).strip()
        if not client_id or not client_secret:
            return False
        try:
            resp = self._request(
                OPENVERSE_TOKEN_API,
                params=None,
                service="openverse",
                method="POST",
                data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
            )
            payload = resp.json()
            token = payload.get("access_token")
            if not token:
                return False
            self.openverse_token = str(token)
            print("Openverse OAuth token acquired.")
            return True
        except Exception as exc:
            print(f"Openverse authentication failed: {exc}", file=sys.stderr)
            return False

    def _wm_ext(self, ext: dict, name: str) -> str:
        raw = ext.get(name, {})
        return clean_html(raw.get("value", "") if isinstance(raw, dict) else str(raw))

    def _candidate_from_wm_page(self, page: dict, query: str, mode: str,
                                category_path: str = "", geo_center_name: str = "") -> Optional[Candidate]:
        infos = page.get("imageinfo") or []
        if not infos:
            return None
        info = infos[0]
        mime = str(info.get("mime", ""))
        if not mime.startswith("image/"):
            return None
        ext = info.get("extmetadata") or {}
        title = page.get("title", "").replace("File:", "", 1)
        desc = self._wm_ext(ext, "ImageDescription")
        return Candidate(
            provider="wikimedia",
            provider_id=str(page.get("pageid", page.get("title", ""))),
            query=query,
            search_mode=mode,
            title=title,
            source_page=info.get("descriptionurl", ""),
            image_url=info.get("url", ""),
            thumbnail_url=info.get("thumburl", "") or info.get("url", ""),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            license=self._wm_ext(ext, "LicenseShortName") or self._wm_ext(ext, "UsageTerms"),
            license_url=self._wm_ext(ext, "LicenseUrl"),
            creator=self._wm_ext(ext, "Artist"),
            attribution=self._wm_ext(ext, "Attribution") or self._wm_ext(ext, "Credit"),
            description=desc,
            source_name="Wikimedia Commons",
            category_path=category_path,
            geo_center_name=geo_center_name,
            guessed_city=guess_city(" ".join([query, title, desc, category_path, geo_center_name])),
        )

    def _wm_generator(self, params: dict, query: str, mode: str,
                      category_path: str = "", geo_center_name: str = "",
                      limit: int = 100) -> Tuple[List[Candidate], bool]:
        found: List[Candidate] = []
        continuation: Optional[dict] = None
        while len(found) < limit:
            req = dict(params)
            req.update({
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiextmetadatafilter": "|".join(LICENSE_FIELDS),
                "iiurlwidth": self.args.thumbnail_width,
            })
            if continuation:
                req.update(continuation)
            try:
                payload = self._request(WIKIMEDIA_API, req, "wikimedia").json()
            except Exception as exc:
                print(f"[{mode}] {query!r} failed after retries: {exc}", file=sys.stderr)
                return found, False
            for page in payload.get("query", {}).get("pages", {}).values():
                c = self._candidate_from_wm_page(page, query, mode, category_path, geo_center_name)
                if c:
                    found.append(c)
                    if len(found) >= limit:
                        break
            continuation = payload.get("continue")
            if not continuation:
                break
        return found[:limit], True

    def search_wikimedia_keyword(self, query: str, limit: int) -> Tuple[List[Candidate], bool]:
        params = {
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit),
        }
        return self._wm_generator(params, query, "wikimedia_keyword", limit=limit)

    def search_wikimedia_geo(self, center: dict, limit: int) -> Tuple[List[Candidate], bool]:
        params = {
            "generator": "geosearch",
            "ggscoord": f"{center['lat']}|{center['lon']}",
            "ggsradius": self.args.geo_radius_m,
            "ggslimit": min(50, limit),
            "ggsnamespace": 6,
        }
        rows, ok = self._wm_generator(
            params, center["name"], "wikimedia_geo",
            geo_center_name=center["name"], limit=limit,
        )
        for c in rows:
            c.geo_lat = center["lat"]
            c.geo_lon = center["lon"]
            c.geo_center_name = center["name"]
            # Dense-grid points use unique names for checkpointing but retain a
            # canonical city label for balancing/reporting.
            c.guessed_city = center.get("city") or c.guessed_city or center["name"]
        return rows, ok

    def _category_subcategories(self, category: str) -> Tuple[List[str], bool]:
        found = []
        cont = None
        while True:
            params = {
                "action": "query", "format": "json", "list": "categorymembers",
                "cmtitle": category, "cmtype": "subcat", "cmlimit": 100,
            }
            if cont:
                params.update(cont)
            try:
                payload = self._request(WIKIMEDIA_API, params, "wikimedia").json()
            except Exception as exc:
                print(f"[category subcats] {category!r} failed: {exc}", file=sys.stderr)
                return found, False
            found.extend(x.get("title", "") for x in payload.get("query", {}).get("categorymembers", []) if x.get("title"))
            cont = payload.get("continue")
            if not cont:
                return found, True

    def _category_files(self, category: str, path: str, limit: int) -> Tuple[List[Candidate], bool]:
        params = {
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmnamespace": 6,
            "gcmlimit": min(50, limit),
        }
        return self._wm_generator(params, category, "wikimedia_category", category_path=path, limit=limit)

    def crawl_wikimedia_category(self, root: str, max_files: int, max_depth: int) -> Tuple[List[Candidate], bool]:
        out: List[Candidate] = []
        queue = [(root, 0, root)]
        visited = set()
        all_ok = True
        while queue and len(out) < max_files:
            category, depth, path = queue.pop(0)
            if category in visited or depth > max_depth:
                continue
            visited.add(category)
            remaining = max_files - len(out)
            rows, ok = self._category_files(category, path, remaining)
            out.extend(rows)
            all_ok = all_ok and ok
            if depth < max_depth:
                subs, sub_ok = self._category_subcategories(category)
                all_ok = all_ok and sub_ok
                for sub in subs:
                    if sub not in visited:
                        queue.append((sub, depth + 1, path + " > " + sub))
            if not ok and not out:
                break
        return out[:max_files], all_ok

    def search_openverse(self, query: str, limit: int) -> Tuple[List[Candidate], bool]:
        if self.openverse_disabled:
            return [], False
        found: List[Candidate] = []
        page = 1
        while len(found) < limit:
            params = {
                "q": query,
                "page": page,
                "page_size": min(50, limit - len(found)),
                "mature": "false",
            }
            # Disabled by default. If user explicitly asks, forward the filter.
            if self.args.openverse_license_type:
                params["license_type"] = self.args.openverse_license_type
            debug = {"query": query, "page": page, "params": dict(params)}
            try:
                resp = self._request(OPENVERSE_API, params, "openverse")
                payload = resp.json()
                debug["status"] = resp.status_code
                debug["results"] = len(payload.get("results", []))
            except Exception as exc:
                debug["error"] = f"{type(exc).__name__}: {exc}"
                self.openverse_debug.append(debug)
                print(f"[openverse] {query!r}: {exc}", file=sys.stderr)
                return found, False
            self.openverse_debug.append(debug)
            results = payload.get("results", [])
            if not results:
                return found, True
            for item in results:
                lic = str(item.get("license") or "")
                ver = str(item.get("license_version") or "")
                if lic and ver:
                    lic = f"{lic}-{ver}"
                title = str(item.get("title") or "")
                tags = item.get("tags") or []
                desc = " ".join(str(x.get("name", "")) if isinstance(x, dict) else str(x) for x in tags)
                found.append(Candidate(
                    provider="openverse",
                    provider_id=str(item.get("id") or ""),
                    query=query,
                    search_mode="openverse_keyword",
                    title=title,
                    source_page=str(item.get("foreign_landing_url") or item.get("detail_url") or ""),
                    image_url=str(item.get("url") or ""),
                    thumbnail_url=str(item.get("thumbnail") or item.get("url") or ""),
                    width=int(item.get("width") or 0),
                    height=int(item.get("height") or 0),
                    license=lic,
                    license_url=str(item.get("license_url") or ""),
                    creator=str(item.get("creator") or ""),
                    attribution=str(item.get("attribution") or ""),
                    description=desc,
                    source_name=str(item.get("source") or "Openverse"),
                    guessed_city=guess_city(" ".join([query, title, desc])),
                ))
                if len(found) >= limit:
                    break
            page_count = int(payload.get("page_count") or page)
            if page >= page_count:
                return found, True
            page += 1
        return found[:limit], True

    def passes_metadata_filter(self, c: Candidate) -> Tuple[bool, str]:
        if not c.provider_id:
            return False, "missing_provider_id"
        if not c.image_url:
            return False, "missing_image_url"
        if c.width and c.width < self.args.min_width:
            return False, f"width<{self.args.min_width}"
        if c.height and c.height < self.args.min_height:
            return False, f"height<{self.args.min_height}"
        if c.width and c.height:
            aspect = c.width / max(c.height, 1)
            if aspect < self.args.min_aspect or aspect > self.args.max_aspect:
                return False, "extreme_aspect_ratio"
        searchable = " ".join([c.title, c.description, c.query, c.category_path]).lower()
        if any(term.lower() in searchable for term in NEGATIVE_TEXT_TERMS):
            return False, "negative_metadata_term"
        key = (c.provider, c.provider_id)
        if key in self.seen_provider_keys:
            return False, "duplicate_provider_id"
        if c.image_url in self.seen_image_urls:
            return False, "duplicate_image_url"
        self.seen_provider_keys.add(key)
        self.seen_image_urls.add(c.image_url)
        return True, ""

    def ingest(self, rows: Sequence[Candidate]):
        for c in rows:
            self.provider_stats[c.provider] += 1
            self.mode_stats[c.search_mode] += 1
            self.query_stats[c.query] += 1
            ok, reason = self.passes_metadata_filter(c)
            city = c.guessed_city or "Unknown"
            quota = int(self.args.max_accepted_per_city)
            if ok and quota > 0 and city not in {"Unknown", "Taiwan"} and self.accepted_city_counts[city] >= quota:
                ok, reason = False, "city_acceptance_cap"
            if ok:
                self.accepted.append(c)
                self.accepted_city_counts[city] += 1
            else:
                c.status = "rejected"
                c.rejection_reason = reason
                self.rejected.append(c)
                self.reject_reason_stats[reason] += 1

    def seed_from_csv(self, path: str):
        """Reuse a previous V2.x candidates.csv without re-querying those images."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Seed CSV not found: {p}")
        added = 0
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    row = dict(row)
                    row["width"] = int(float(row.get("width") or 0))
                    row["height"] = int(float(row.get("height") or 0))
                    for key in ("geo_lat", "geo_lon", "clip_street_probability"):
                        v = row.get(key)
                        row[key] = float(v) if v not in (None, "", "None") else None
                    # Keep only current dataclass fields for compatibility with older V2.x CSVs.
                    fields = Candidate.__dataclass_fields__
                    row = {k: row.get(k, fields[k].default if fields[k].default is not None else "") for k in fields}
                    c = Candidate(**row)
                except Exception as exc:
                    print(f"[seed] skip malformed row: {exc}", file=sys.stderr)
                    continue
                key = (c.provider, c.provider_id)
                if key in self.seen_provider_keys or (c.image_url and c.image_url in self.seen_image_urls):
                    continue
                c.status = "candidate"
                self.accepted.append(c)
                self.seen_provider_keys.add(key)
                if c.image_url:
                    self.seen_image_urls.add(c.image_url)
                self.accepted_city_counts[c.guessed_city or "Unknown"] += 1
                added += 1
        self.seed_count += added
        print(f"Seed CSV loaded: {p} | added={added} | accepted total={len(self.accepted)}")
        self.save_checkpoint()

    def run_unit(self, unit_id: str, label: str, fn):
        if unit_id in self.completed_units:
            print(f"[resume] skip completed: {label}")
            return
        if len(self.accepted) >= self.args.target_candidates:
            return
        print(f"\n=== {label} ===")
        rows, ok = fn()
        print(f"  raw={len(rows)}")
        self.ingest(rows)
        print(f"  accepted total={len(self.accepted)} | rejected total={len(self.rejected)}")
        if ok:
            self.completed_units.add(unit_id)
        else:
            print("  unit incomplete; it will be retried with --resume")
        self.save_checkpoint()

    def discover(self, keyword_queries: Sequence[str], categories: Sequence[str], geo_centers: Sequence[dict]):
        # Diversity-first order: round-robin geo grid -> categories -> keywords -> Openverse keywords.
        if self.args.enable_geo_search and "wikimedia" in self.args.providers:
            for center in geo_centers:
                self.run_unit(
                    f"geo:{center['name']}", f"Geo Search: {center['name']}",
                    lambda c=center: self.search_wikimedia_geo(c, self.args.max_per_geo_center),
                )
                if len(self.accepted) >= self.args.target_candidates:
                    break

        if self.args.enable_category_search and "wikimedia" in self.args.providers and len(self.accepted) < self.args.target_candidates:
            for category in categories:
                self.run_unit(
                    f"category:{category}", f"Category Crawl: {category}",
                    lambda c=category: self.crawl_wikimedia_category(c, self.args.max_per_category, self.args.category_depth),
                )
                if len(self.accepted) >= self.args.target_candidates:
                    break

        if self.args.enable_keyword_search and "wikimedia" in self.args.providers and len(self.accepted) < self.args.target_candidates:
            for q in keyword_queries:
                self.run_unit(
                    f"wm_keyword:{q}", f"Wikimedia Keyword: {q}",
                    lambda x=q: self.search_wikimedia_keyword(x, self.args.max_per_keyword_query),
                )
                if len(self.accepted) >= self.args.target_candidates:
                    break

        if self.args.enable_keyword_search and "openverse" in self.args.providers and len(self.accepted) < self.args.target_candidates:
            for q in keyword_queries:
                if self.openverse_disabled:
                    print("Openverse disabled for this run after authentication failure; skipping remaining Openverse queries.")
                    break
                self.run_unit(
                    f"ov_keyword:{q}", f"Openverse Keyword: {q}",
                    lambda x=q: self.search_openverse(x, self.args.max_per_keyword_query),
                )
                if len(self.accepted) >= self.args.target_candidates:
                    break

        return self.accepted[:self.args.target_candidates], self.rejected

    def download(self, candidates: List[Candidate]):
        image_dir = self.out_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        sha_seen: Dict[str, str] = {}
        dhash_seen: Dict[str, str] = {}
        for idx, c in enumerate(candidates, 1):
            if c.status in {"accepted", "downloaded"} and c.downloaded_path:
                continue
            print(f"[download {idx}/{len(candidates)}] {c.provider} {c.title[:70]}")
            try:
                # Image hosts have their own limits. Basic retry is enough here.
                resp = self._request(c.image_url, None, "wikimedia" if c.provider == "wikimedia" else "openverse", timeout=90)
                data = resp.content
                sha = hashlib.sha256(data).hexdigest()
                c.sha256 = sha
                if sha in sha_seen:
                    c.status = "rejected"; c.rejection_reason = f"exact_duplicate_of:{sha_seen[sha]}"; continue
                image = Image.open(io.BytesIO(data)); image.load(); image = image.convert("RGB")
                if image.width < self.args.min_width or image.height < self.args.min_height:
                    c.status = "rejected"; c.rejection_reason = "downloaded_image_too_small"; continue
                dh = image_dhash(image); c.dhash = dh
                if dh in dhash_seen:
                    c.status = "rejected"; c.rejection_reason = f"perceptual_duplicate_of:{dhash_seen[dh]}"; continue
                ext = infer_extension(c.image_url, image.format)
                filename = slugify(f"candidate_{idx:05d}_{c.provider}_{c.guessed_city or c.geo_center_name or 'unknown'}")
                path = image_dir / f"{filename}{ext}"
                if ext in {".jpg", ".jpeg"}:
                    image.save(path, quality=95, subsampling=0)
                elif ext == ".png":
                    image.save(path)
                else:
                    path = path.with_suffix(".jpg"); image.save(path, quality=95, subsampling=0)
                c.downloaded_path = str(path.relative_to(self.out_dir))
                c.status = "downloaded"
                sha_seen[sha] = c.downloaded_path
                dhash_seen[dh] = c.downloaded_path
            except Exception as exc:
                c.status = "rejected"; c.rejection_reason = f"download_error:{type(exc).__name__}"
                print(f"  download failed: {exc}", file=sys.stderr)
        self.save_checkpoint()


def image_dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for y in range(hash_size):
        row = pixels[y*(hash_size+1):(y+1)*(hash_size+1)]
        bits.extend(row[x] > row[x+1] for x in range(hash_size))
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{math.ceil(len(bits)/4)}x}"


def hamming_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 999


def infer_extension(url: str, pil_format: Optional[str]) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get((pil_format or "").upper(), ".jpg")


def run_clip_filter(candidates: List[Candidate], out_dir: Path, args):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=args.clip_local_only)
    model = CLIPModel.from_pretrained(args.clip_model, local_files_only=args.clip_local_only).to(device).eval()
    texts = POSITIVE_CLIP_TEXTS + NEGATIVE_CLIP_TEXTS
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        if not torch.is_tensor(text_features) and hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    for c in candidates:
        if c.status != "downloaded" or not c.downloaded_path:
            continue
        try:
            image = Image.open(out_dir / c.downloaded_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                image_features = model.get_image_features(**image_inputs)
                if not torch.is_tensor(image_features) and hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                probs = torch.softmax((image_features @ text_features.T)[0] * 100.0, dim=0)
            p = float(probs[:len(POSITIVE_CLIP_TEXTS)].sum().cpu())
            c.clip_street_probability = p
            if p < args.clip_min_probability:
                c.status = "rejected"; c.rejection_reason = f"clip_street_probability<{args.clip_min_probability:.2f}"
            else:
                c.status = "accepted"
        except Exception as exc:
            c.status = "review"; c.rejection_reason = f"clip_error:{type(exc).__name__}"
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_csv(path: Path, rows: Sequence[Candidate]):
    fields = list(Candidate.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for x in rows:
            w.writerow(asdict(x))


def write_near_duplicates(path: Path, rows: Sequence[Candidate], max_distance: int):
    xs = [x for x in rows if x.dhash]
    pairs = []
    for i, a in enumerate(xs):
        for b in xs[i+1:]:
            d = hamming_hex(a.dhash, b.dhash)
            if d <= max_distance:
                pairs.append({"a": a.downloaded_path, "b": b.downloaded_path, "distance": d,
                              "a_source": a.source_page, "b_source": b.source_page})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["a", "b", "distance", "a_source", "b_source"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pairs)


def load_lines(path: Optional[str]) -> List[str]:
    if not path:
        return []
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]


def stable_unique(xs: Sequence[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def generate_geo_grid(base_centers: Sequence[dict], levels: int) -> List[dict]:
    """Generate interleaved city grid points: all city centers, then all N, E, ... points."""
    levels = max(1, min(int(levels), len(GEO_GRID_OFFSETS)))
    points = []
    for offset_name, dx_units, dy_units in GEO_GRID_OFFSETS[:levels]:
        for base in base_centers:
            city = base["name"]
            step_km = MAJOR_GRID_STEP_KM.get(city, MEDIUM_GRID_STEP_KM.get(city, 2.8))
            lat = float(base["lat"])
            lon = float(base["lon"])
            dy_km = dy_units * step_km
            dx_km = dx_units * step_km
            point_lat = lat + dy_km / 111.32
            lon_scale = max(20.0, 111.32 * math.cos(math.radians(lat)))
            point_lon = lon + dx_km / lon_scale
            points.append({
                "name": f"{city}_{offset_name}",
                "city": city,
                "lat": round(point_lat, 6),
                "lon": round(point_lon, 6),
                "grid_offset": offset_name,
                "grid_step_km": step_km,
            })
    return points


def load_geo(args) -> List[dict]:
    if args.geo_centers_json:
        return json.loads(Path(args.geo_centers_json).read_text(encoding="utf-8"))
    if args.geo_grid:
        return generate_geo_grid(DEFAULT_GEO_CENTERS, args.geo_grid_levels)
    return [dict(x, city=x["name"]) for x in DEFAULT_GEO_CENTERS]


def write_reports(out_dir: Path, finder: Finder, accepted, rejected, args):
    active = [x for x in accepted if x.status in {"candidate", "downloaded", "accepted"}]
    city_counts = Counter((x.guessed_city or x.geo_center_name or "Unknown") for x in active)
    provider_counts = Counter(x.provider for x in active)
    mode_counts = Counter(x.search_mode for x in active)
    license_counts = Counter(x.license or "Unknown" for x in active)
    report = ["# Corpus Diversity Report", "", f"Active candidates: {len(active)}", "", "## Provider"]
    report += [f"- {k}: {v}" for k, v in provider_counts.most_common()]
    report += ["", "## Search mode"] + [f"- {k}: {v}" for k, v in mode_counts.most_common()]
    report += ["", "## Guessed city"] + [f"- {k}: {v}" for k, v in city_counts.most_common()]
    report += ["", "## License"] + [f"- {k}: {v}" for k, v in license_counts.most_common(30)]
    (out_dir / "CORPUS_DIVERSITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    diagnostics = {
        "version": VERSION,
        "http_status_counts": dict(finder.http_status_stats),
        "retry_events": finder.retry_events,
        "provider_result_counts": dict(finder.provider_stats),
        "search_mode_counts": dict(finder.mode_stats),
        "reject_reason_counts": dict(finder.reject_reason_stats),
        "completed_units": sorted(finder.completed_units),
        "openverse_disabled": finder.openverse_disabled,
        "openverse_debug": finder.openverse_debug,
        "seed_count": finder.seed_count,
        "accepted_city_counts": dict(finder.accepted_city_counts),
        "max_accepted_per_city": args.max_accepted_per_city,
        "geo_grid": args.geo_grid,
        "geo_grid_levels": args.geo_grid_levels,
    }
    (out_dir / "provider_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "version": VERSION,
        "metadata_candidates": len(accepted),
        "final_candidate_count": len(active),
        "rejected_or_review_count": len([x for x in accepted + rejected if x.status in {"rejected", "review"}]),
        "target_candidates": args.target_candidates,
        "download": args.download,
        "clip_filter": args.clip_filter,
        "checkpoint": str(finder.checkpoint_path),
        "seed_count": finder.seed_count,
        "geo_points": len(load_geo(args)),
        "max_accepted_per_city": args.max_accepted_per_city,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Taiwan Streetscape Finder V2.2")
    p.add_argument("--output", default="taiwan_streetscape_candidates_v2_1")
    p.add_argument("--providers", nargs="+", choices=["wikimedia", "openverse"], default=["wikimedia", "openverse"])
    p.add_argument("--target-candidates", type=int, default=3000)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--enable-keyword-search", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-category-search", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-geo-search", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keyword-queries", nargs="*")
    p.add_argument("--keyword-query-file")
    p.add_argument("--max-per-keyword-query", type=int, default=80)
    p.add_argument("--categories", nargs="*")
    p.add_argument("--category-file")
    p.add_argument("--max-per-category", type=int, default=300)
    p.add_argument("--category-depth", type=int, default=2)
    p.add_argument("--geo-centers-json")
    p.add_argument("--geo-grid", action=argparse.BooleanOptionalAction, default=True,
                   help="Generate an interleaved multi-point grid around each Taiwanese city (default true).")
    p.add_argument("--geo-grid-levels", type=int, default=9,
                   help="Number of ordered grid offsets per city; 9 = center + cardinal + diagonals.")
    p.add_argument("--max-per-geo-center", type=int, default=45,
                   help="Per grid-point metadata cap; intentionally modest for diversity.")
    p.add_argument("--geo-radius-m", type=int, default=3500,
                   help="Geo search radius for each grid point.")
    p.add_argument("--max-accepted-per-city", type=int, default=220,
                   help="Cap newly accepted items per canonical city; 0 disables the cap.")
    p.add_argument("--seed-csv", default="",
                   help="Optional prior V2.x candidates.csv to reuse as initial candidates.")

    p.add_argument("--min-width", type=int, default=900)
    p.add_argument("--min-height", type=int, default=600)
    p.add_argument("--min-aspect", type=float, default=0.60)
    p.add_argument("--max-aspect", type=float, default=2.40)
    p.add_argument("--thumbnail-width", type=int, default=640)

    p.add_argument("--wikimedia-contact", default="", help="Real contact email/URL for Wikimedia User-Agent, e.g. mailto:name@example.com")
    p.add_argument("--wikimedia-request-delay", type=float, default=1.0)
    p.add_argument("--openverse-request-delay", type=float, default=2.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--backoff-base", type=float, default=5.0)
    p.add_argument("--max-retry-wait", type=float, default=600.0)

    p.add_argument("--openverse-license-type", default="", help="Optional Openverse license_type filter; empty by default.")
    p.add_argument("--openverse-client-id", default="")
    p.add_argument("--openverse-client-secret", default="")

    p.add_argument("--download", action="store_true")
    p.add_argument("--clip-filter", action="store_true")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip-local-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-min-probability", type=float, default=0.55)
    p.add_argument("--near-duplicate-distance", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    keywords = stable_unique(
        list(args.keyword_queries or []) + load_lines(args.keyword_query_file) +
        ([] if args.keyword_queries or args.keyword_query_file else DEFAULT_KEYWORD_QUERIES)
    )
    categories = stable_unique(
        list(args.categories or []) + load_lines(args.category_file) +
        ([] if args.categories or args.category_file else DEFAULT_CATEGORIES)
    )
    geo_centers = load_geo(args)

    print(f"Taiwan Streetscape Finder V{VERSION}")
    print(f"Providers: {args.providers}")
    print(f"Search order: round-robin geo grid -> category -> keyword -> Openverse")
    print(f"Geo points: {len(geo_centers)} | city cap: {args.max_accepted_per_city}")
    print(f"Target candidates: {args.target_candidates}")

    finder = Finder(args, out_dir)
    if args.seed_csv and not args.resume:
        finder.seed_from_csv(args.seed_csv)
    accepted, rejected = finder.discover(keywords, categories, geo_centers)

    if args.download:
        finder.download(accepted)
    if args.clip_filter:
        if not args.download:
            raise SystemExit("--clip-filter requires --download")
        run_clip_filter(accepted, out_dir, args)
        finder.save_checkpoint()

    final_accepted = [x for x in accepted if x.status in {"candidate", "downloaded", "accepted"}]
    final_rejected = [x for x in accepted + rejected if x.status in {"rejected", "review"}]
    write_csv(out_dir / "candidates.csv", final_accepted)
    write_csv(out_dir / "rejected.csv", final_rejected)
    write_csv(out_dir / "all_results.csv", accepted + rejected)
    if args.download:
        write_near_duplicates(out_dir / "near_duplicates.csv", accepted, args.near_duplicate_distance)
    summary = write_reports(out_dir, finder, accepted, rejected, args)

    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
