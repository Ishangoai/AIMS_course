import json
import urllib.parse
import urllib.request
from typing import Any, Dict

UA = "ArticleWriterBot/0.1 (contact: you@example.com) Python-urllib"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
    "Referer": "https://en.wikipedia.org/",
}


class WikipediaSearcher:
    def __init__(self, lang: str = "en", timeout: int = 15, ua: str = UA, verbose: bool = True):
        self.lang, self.timeout, self.ua, self.verbose = lang, timeout, ua, verbose
        self.hdrs = {
            "User-Agent": ua,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
            "Referer": "https://en.wikipedia.org/"
        }
        self.enabled = True

    def _get_json(self, url: str, params: Dict[str, Any] | None = None):
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self.hdrs, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def search(self, query: str, num_results: int = 5):
        try:
            # Action API search + extracts
            base = f"https://{self.lang}.wikipedia.org/w/api.php"
            data = self._get_json(base, {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(1, num_results),
                "format": "json", "utf8": 1, "formatversion": 2})
            hits = (data.get("query") or {}).get("search") or []
            if not hits:
                return []
            pageids = [str(h["pageid"]) for h in hits]
            detail = self._get_json(base, {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
                "pageids": "|".join(pageids),
                "format": "json",
                "utf8": 1, "formatversion": 2})
            pages = (detail.get("query") or {}).get("pages") or []
            out = []
            for p in pages[:num_results]:
                title = p.get("title", "")
                extract = (p.get("extract") or "").strip()
                pid = p.get("pageid")
                link = f"https://{self.lang}.wikipedia.org/?curid={pid}" if pid else ""
                out.append({"title": title, "link": link, "content": extract, "engine": "wikipedia"})
            return out
        except Exception:
            # REST fallback: summary of first search hit
            try:
                base = f"https://{self.lang}.wikipedia.org/w/api.php"
                data = self._get_json(base, {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                    "utf8": 1,
                    "formatversion": 2
                })
                hits = (data.get("query") or {}).get("search") or []
                if not hits:
                    return []
                title = hits[0]["title"]
                safe = urllib.parse.quote(title.replace(" ", "_"))
                rest = f"https://{self.lang}.wikipedia.org/api/rest_v1/page/summary/{safe}"
                s = self._get_json(rest, None)
                return [{
                    "title": s.get("title", title),
                    "link": s.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "content": (s.get("extract") or "").strip(),
                    "engine": "wikipedia-rest"
                }]
            except Exception:
                return []
