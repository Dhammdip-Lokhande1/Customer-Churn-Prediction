import os
import sys
import ssl
import urllib.request
import hashlib

# ─── Dataset Source ──────────────────────────────────────────────────────────
# Public IBM Telco Customer Churn dataset (Kaggle mirror, no authentication required)
DATASET_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code"
    "/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# SHA-256 checksum of the expected file — used to verify download integrity.
# Update this value if the upstream file ever changes.
EXPECTED_SHA256 = "ea92973a0e5aa7be76a30c8de9b56c79e5d3eef9e9e5ce4a06bba01cd01cba45"

# Resolve destination relative to this script's location (robust to CWD changes)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEST_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
DEST_PATH = os.path.join(DEST_DIR, "telco_churn.csv")


def _sha256(path: str) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset() -> None:
    """
    Download the IBM Telco Customer Churn CSV dataset to data/raw/.

    Security measures applied:
    - HTTPS with system-verified TLS certificates (no ssl.CERT_NONE).
    - SHA-256 checksum verification after download.
    - Partial download is cleaned up on failure.
    """
    os.makedirs(DEST_DIR, exist_ok=True)

    if os.path.exists(DEST_PATH):
        print(f"[INFO] Dataset already present at: {DEST_PATH}")
        print("[INFO] Verifying checksum …")
        digest = _sha256(DEST_PATH)
        if digest == EXPECTED_SHA256:
            print("[OK]   Checksum verified — dataset is intact.")
        else:
            print(
                f"[WARN] Checksum mismatch!\n"
                f"       Expected : {EXPECTED_SHA256}\n"
                f"       Got      : {digest}\n"
                f"       The local file may be corrupted or outdated.\n"
                f"       Delete {DEST_PATH} and re-run to refresh."
            )
        return

    # Build a secure SSL context (verifies server certificate chain)
    ssl_ctx = ssl.create_default_context()

    print(f"[INFO] Downloading dataset from:\n       {DATASET_URL}")
    print(f"[INFO] Destination: {DEST_PATH}")

    try:
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "ChurnSight-DataDownloader/1.0"},
        )
        with urllib.request.urlopen(req, context=ssl_ctx) as response, \
                open(DEST_PATH, "wb") as out_file:
            out_file.write(response.read())

        print("[INFO] Download complete. Verifying checksum …")
        digest = _sha256(DEST_PATH)
        if digest != EXPECTED_SHA256:
            os.remove(DEST_PATH)
            raise RuntimeError(
                f"Checksum verification FAILED!\n"
                f"  Expected : {EXPECTED_SHA256}\n"
                f"  Got      : {digest}\n"
                f"  The downloaded file has been deleted for safety.\n"
                f"  This may indicate a man-in-the-middle attack or a "
                f"changed upstream file."
            )

        print("[OK]   Checksum verified — dataset downloaded and intact.")

    except Exception as exc:
        # Clean up partial download if anything went wrong
        if os.path.exists(DEST_PATH):
            os.remove(DEST_PATH)
        print(f"[ERROR] Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
