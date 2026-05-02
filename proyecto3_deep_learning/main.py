from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))

from app import demo, get_agent


def main() -> None:
    get_agent()
    demo.launch()


if __name__ == "__main__":
    main()
