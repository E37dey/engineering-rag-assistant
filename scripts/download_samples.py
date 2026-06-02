"""One-shot download of sample engineering datasheets into data/.

Pulls classic analog/mixed-signal parts directly from manufacturer sites.
Used to populate data/ with realistic test material for the RAG pipeline.

Run: python scripts/download_samples.py
"""
import sys
import urllib.request
from pathlib import Path

# Windows consoles often default to cp1252 — force UTF-8 so the script's
# diagnostic output renders correctly on every platform.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path(__file__).parent.parent / "data"

# Some manufacturer CDNs reject the default Python User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Each part lists candidate URLs in priority order — first success wins.
# All sources are official manufacturer sites (the part is genuinely
# second-sourced when multiple vendors appear).
DATASHEETS = [
    {
        # AD620 was the original target but analog.com's WAF blocks scripted
        # downloads (TLS fingerprint, not User-Agent). Substituting INA128 —
        # TI's pin-compatible instrumentation amplifier in the same role.
        "name": "INA128.pdf",
        "description": "TI INA128 instrumentation amplifier (substituted for AD620)",
        "urls": [
            "https://www.ti.com/lit/ds/symlink/ina128.pdf",
        ],
    },
    {
        "name": "LM358.pdf",
        "description": "LM358 dual op-amp (TI / ST / onsemi)",
        "urls": [
            "https://www.ti.com/lit/ds/symlink/lm358.pdf",
            "https://www.st.com/resource/en/datasheet/lm358.pdf",
            "https://www.onsemi.com/pdf/datasheet/lm358-d.pdf",
        ],
    },
    {
        "name": "NE555.pdf",
        "description": "NE555 precision timer (TI / ST / onsemi)",
        "urls": [
            "https://www.ti.com/lit/ds/symlink/ne555.pdf",
            "https://www.st.com/resource/en/datasheet/ne555.pdf",
            "https://www.onsemi.com/pdf/datasheet/ne555-d.pdf",
        ],
    },
    {
        "name": "MAX232.pdf",
        "description": "Analog Devices (Maxim) MAX232 RS-232 driver/receiver",
        "urls": [
            "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX220-MAX249.pdf",
            "https://www.ti.com/lit/ds/symlink/max232.pdf",
        ],
    },
]


def download(url: str, dest: Path) -> int:
    """Download `url` to `dest`. Returns bytes written. Raises on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(
            f"response does not look like a PDF (starts with {data[:8]!r})"
        )
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    successes = 0

    for sheet in DATASHEETS:
        name = sheet["name"]
        dest = DATA_DIR / name

        if dest.exists():
            size = dest.stat().st_size
            print(f"[skip]  {name} already present ({size:,} bytes)")
            successes += 1
            continue

        print(f"[get]   {name} - {sheet['description']}")
        downloaded = False
        for url in sheet["urls"]:
            try:
                size = download(url, dest)
                print(f"[ok]    {name} - {size:,} bytes from {url}")
                successes += 1
                downloaded = True
                break
            # Broad catch on purpose: any network/SSL/timeout/HTTP/validation
            # failure should fall through to the next candidate URL, not
            # crash the script.
            except Exception as e:
                print(f"[fail]  {url} -> {type(e).__name__}: {e}")
        if not downloaded:
            print(f"[err]   {name} - all sources failed")

    print()
    print(f"Downloaded {successes}/{len(DATASHEETS)} datasheets into {DATA_DIR.resolve()}")

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if pdfs:
        print()
        print("Files in data/:")
        for pdf in pdfs:
            size = pdf.stat().st_size
            print(f"  {pdf.name:<14} {size:>10,} bytes  ({size / 1024:,.1f} KB)")

    return 0 if successes == len(DATASHEETS) else 1


if __name__ == "__main__":
    sys.exit(main())
