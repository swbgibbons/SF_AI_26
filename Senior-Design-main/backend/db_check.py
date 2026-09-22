import os
import psycopg
import traceback


def main() -> None:
    cfg = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS"),
        # Default to "prefer" so it works with non-SSL local DBs; override with DB_SSLMODE
        "sslmode": os.getenv("DB_SSLMODE", "prefer"),
        "connect_timeout": 5,
    }

    redacted_cfg = {k: ("***" if k == "password" else v)
                    for k, v in cfg.items()}
    print("[DB CHECK] Config:", redacted_cfg)

    try:
        with psycopg.connect(**cfg) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
        print("[DB CHECK] Connected OK:", version[0]
              if version else "(no version)")
    except Exception as exc:  # noqa: BLE001
        print("[DB CHECK] Connect failed:", exc)
        traceback.print_exc()


if __name__ == "__main__":
    main()
