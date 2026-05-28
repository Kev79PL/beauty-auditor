import os
import json
import re
import time
import threading
import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
import stripe

GEMINI_MODEL = "gemini-2.5-flash"

app = Flask(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def normalize(url):
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def fetch_website(url):
    resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title else ""

    meta_desc = ""
    tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if tag:
        meta_desc = tag.get("content", "")

    meta_kw = ""
    tag = soup.find("meta", attrs={"name": re.compile("^keywords$", re.I)})
    if tag:
        meta_kw = tag.get("content", "")

    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        t = h.get_text(strip=True)
        if t and len(t) < 200:
            headings.append(t)

    social = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        for p in ("instagram", "facebook", "tiktok", "youtube", "pinterest"):
            if p in href and a["href"] not in social:
                social.append(a["href"])

    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    body = " ".join(soup.get_text(" ", strip=True).split())[:5000]

    out = f"URL: {url}\nTytuł: {title}\nMeta opis: {meta_desc}\n"
    if meta_kw:
        out += f"Meta keywords: {meta_kw}\n"
    if headings:
        out += f"Nagłówki: {' | '.join(headings[:20])}\n"
    if social:
        out += f"Linki social media: {', '.join(social[:5])}\n"
    out += f"\nTreść strony:\n{body}"
    return out


SITE_PROMPT = (
    "Jesteś przyjaznym doradcą marketingowym dla właścicielek gabinetów kosmetycznych i beauty. "
    "Piszesz prostym, zrozumiałym językiem — bez technicznego żargonu. "
    "Zwróć WYŁĄCZNIE surowy obiekt JSON (bez markdown, bez backticks).\n\n"
    "Struktura JSON:\n"
    "{\n"
    '  "brand_name": "nazwa firmy lub gabinetu",\n'
    '  "website_score": 0,\n'
    '  "seo_score": 0,\n'
    '  "offers_score": 0,\n'
    '  "keywords": {\n'
    '    "primary": ["3-5 fraz na które strona się pozycjonuje"],\n'
    '    "secondary": ["3-5 fraz dodatkowych"],\n'
    '    "missing": ["3-5 ważnych fraz których brakuje"]\n'
    '  },\n'
    '  "findings": {\n'
    '    "good": ["2-3 mocne strony tej witryny — konkretnie i prosto"],\n'
    '    "warn": ["2-3 rzeczy do poprawy"],\n'
    '    "bad": ["1-2 wyraźne słabości"]\n'
    '  }\n'
    "}\n\n"
    "Zasady: score = liczby całkowite 0-100. Cały tekst po polsku. "
    "Pisz tak, żeby właścicielka gabinetu beauty od razu rozumiała o co chodzi — "
    "nie używaj słów takich jak 'meta tagi', 'crawl', 'indeksacja'. "
    "Zamiast tego pisz np. 'strona jest trudna do znalezienia w Google'."
)

COMPARE_PROMPT = (
    "Jesteś doradcą marketingowym dla właścicielek gabinetów beauty. "
    "Porównaj wyniki analizy strony KONKURENCJI i MOJEJ strony. "
    "Zwróć WYŁĄCZNIE surowy JSON (bez markdown, bez backticks).\n\n"
    "Struktura JSON:\n"
    "{\n"
    '  "losing": [\n'
    '    "konkretny punkt: w czym moja strona jest słabsza od konkurencji",\n'
    '    "np. Konkurencja ma czytelniejszy cennik usług na stronie głównej"\n'
    '  ],\n'
    '  "winning": [\n'
    '    "konkretny punkt: w czym moja strona jest lepsza",\n'
    '    "np. Twoja strona lepiej opisuje dostępne zabiegi"\n'
    '  ],\n'
    '  "actions": [\n'
    '    "Bardzo konkretna akcja 1 do wykonania w tym tygodniu",\n'
    '    "Bardzo konkretna akcja 2",\n'
    '    "Bardzo konkretna akcja 3"\n'
    '  ]\n'
    "}\n\n"
    "WAŻNE dla 'losing' i 'winning': 2-4 punkty każda lista. "
    "WAŻNE dla 'actions': 3 BARDZO KONKRETNE zadania. "
    "Zamiast 'popraw SEO' pisz 'Dodaj frazę [konkretna fraza] do tytułu strony głównej'. "
    "Zamiast 'dodaj treści' pisz 'Stwórz sekcję z opiniami klientek — minimum 5 opinii z imieniem'. "
    "Wszystko po polsku, językiem właścicielki gabinetu."
)


class GeminiUnavailableError(Exception):
    pass


def _call_gemini(client, system, user_msg, max_tokens=4096):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            if "503" in msg or "UNAVAILABLE" in msg:
                if attempt < 2:
                    time.sleep(3)
                    continue
                raise GeminiUnavailableError()
            raise


def _parse_json(text, default=None):
    if not text:
        return default if default is not None else {}
    for pat in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break

    return default if default is not None else {}


def _validate_site(r):
    for k, v in {
        "brand_name": "Nieznana firma",
        "website_score": 50,
        "seo_score": 50,
        "offers_score": 50,
        "keywords": {"primary": [], "secondary": [], "missing": []},
        "findings": {"good": [], "warn": [], "bad": []},
    }.items():
        r.setdefault(k, v)
    if not isinstance(r.get("keywords"), dict):
        r["keywords"] = {"primary": [], "secondary": [], "missing": []}
    for k in ("primary", "secondary", "missing"):
        if not isinstance(r["keywords"].get(k), list):
            r["keywords"][k] = []
    if not isinstance(r.get("findings"), dict):
        r["findings"] = {"good": [], "warn": [], "bad": []}
    for k in ("good", "warn", "bad"):
        if not isinstance(r["findings"].get(k), list):
            r["findings"][k] = []
    for s in ("website_score", "seo_score", "offers_score"):
        r[s] = max(0, min(100, int(r.get(s, 50))))
    return r


def _validate_comparison(r):
    for k in ("losing", "winning", "actions"):
        if not isinstance(r.get(k), list):
            r[k] = []
    r["actions"] = r["actions"][:3]
    return r


def analyze_site(website_text, api_key):
    client = genai.Client(api_key=api_key)
    text = _call_gemini(client, SITE_PROMPT, f"Przeanalizuj tę stronę:\n\n{website_text}")
    return _validate_site(_parse_json(text))


def compare_sites(competitor, mine, api_key):
    try:
        client = genai.Client(api_key=api_key)
        user_msg = (
            f"Dane strony KONKURENCJI:\n{json.dumps(competitor, ensure_ascii=False)}\n\n"
            f"Dane MOJEJ strony:\n{json.dumps(mine, ensure_ascii=False)}"
        )
        text = _call_gemini(client, COMPARE_PROMPT, user_msg, max_tokens=1000)
        return _validate_comparison(_parse_json(text, default={"losing": [], "winning": [], "actions": []}))
    except Exception:
        return {"losing": [], "winning": [], "actions": []}


N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")
N8N_WEBHOOK_PLATNOSC_URL = os.environ.get("N8N_WEBHOOK_PLATNOSC_URL", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")


def _send_webhook(payload, url=None):
    target = url or N8N_WEBHOOK_URL
    if not target:
        print(f"[WEBHOOK] (no URL configured) {json.dumps(payload, ensure_ascii=False)}")
        return
    try:
        requests.post(target, json=payload, timeout=10)
    except Exception as e:
        print(f"[WEBHOOK] Error: {e}")


def send_webhook_async(payload, url=None):
    threading.Thread(target=_send_webhook, args=(payload,), kwargs={"url": url}, daemon=True).start()


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    competitor_url = normalize(data.get("competitor", ""))
    my_url = normalize(data.get("mine", ""))

    if not competitor_url or not my_url:
        return jsonify({"error": "Obie adresy stron są wymagane"}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY nie jest ustawiony"}), 500

    client = genai.Client(api_key=api_key)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_comp = ex.submit(fetch_website, competitor_url)
        fut_mine = ex.submit(fetch_website, my_url)
        try:
            competitor_text = fut_comp.result(timeout=30)
        except Exception as e:
            return jsonify({"error": f"Nie można pobrać strony konkurencji: {e}"}), 500
        try:
            my_text = fut_mine.result(timeout=30)
        except Exception as e:
            return jsonify({"error": f"Nie można pobrać Twojej strony: {e}"}), 500

    _unavailable_msg = "Analiza chwilowo niedostępna — spróbuj ponownie za chwilę"

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_comp_ana = ex.submit(analyze_site, competitor_text, api_key)
        fut_mine_ana = ex.submit(analyze_site, my_text, api_key)
        try:
            competitor_result = fut_comp_ana.result(timeout=120)
        except GeminiUnavailableError:
            return jsonify({"error": _unavailable_msg}), 503
        except Exception as e:
            return jsonify({"error": f"Błąd analizy strony konkurencji: {type(e).__name__}: {e}"}), 500
        try:
            my_result = fut_mine_ana.result(timeout=120)
        except GeminiUnavailableError:
            return jsonify({"error": _unavailable_msg}), 503
        except Exception as e:
            return jsonify({"error": f"Błąd analizy Twojej strony: {type(e).__name__}: {e}"}), 500

    try:
        comparison = compare_sites(competitor_result, my_result, api_key)
    except GeminiUnavailableError:
        comparison = {"losing": [], "winning": [], "actions": []}
    except Exception as e:
        comparison = {"losing": [], "winning": [], "actions": [], "error": str(e)}

    return jsonify({
        "competitor": competitor_result,
        "mine": my_result,
        "comparison": comparison,
    })


@app.route("/zamow")
def zamow():
    return send_file("zamow.html")


@app.route("/dziekujemy")
def dziekujemy():
    return send_file("dziekujemy.html")


@app.route("/api/save-order", methods=["POST"])
def save_order():
    data = request.get_json() or {}
    payload = {
        "imie_firma": data.get("imie_firma", ""),
        "nip": data.get("nip", ""),
        "ulica": data.get("ulica", ""),
        "kod_pocztowy": data.get("kod_pocztowy", ""),
        "miasto": data.get("miasto", ""),
        "kod_miasto": data.get("kod_pocztowy", "") + " " + data.get("miasto", ""),
        "email": data.get("email", ""),
        "url_strony": data.get("url_strony", ""),
        "url_facebook": data.get("url_facebook", ""),
        "url_instagram": data.get("url_instagram", ""),
        "url_konkurent_1": data.get("url_konkurent_1", ""),
        "url_konkurent_2": data.get("url_konkurent_2", ""),
        "url_konkurent_3": data.get("url_konkurent_3", ""),
        "status": "oczekuje_na_platnosc",
        "kwota": "99",
        "data_zamowienia": datetime.now(timezone.utc).isoformat(),
        "zrodlo": "audyt.spacepr.pl",
    }
    send_webhook_async(payload, url=N8N_WEBHOOK_PLATNOSC_URL)
    return jsonify({"success": True})


@app.route("/api/save-lead", methods=["POST"])
def save_lead():
    data = request.get_json() or {}
    payload = {
        "konkurent": data.get("konkurent", ""),
        "moja_strona": data.get("moja_strona", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "typ": "darmowy_audyt",
    }
    send_webhook_async(payload)
    return jsonify({"success": True})


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        amount = session.get("amount_total") or 0
        created = session.get("created") or 0
        n8n_payload = {
            "email": (session.get("customer_details") or {}).get("email", ""),
            "status": "oplacone",
            "stripe_session_id": session.get("id", ""),
            "kwota": str(amount // 100),
            "data_platnosci": datetime.fromtimestamp(created, tz=timezone.utc).isoformat(),
            "zrodlo": "stripe_webhook",
        }
        send_webhook_async(n8n_payload, url=N8N_WEBHOOK_PLATNOSC_URL)

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Running on http://localhost:5001")
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
