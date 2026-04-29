from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_main():
    try:
        from person_yolo.web.app import main as app_main

        return app_main
    except ModuleNotFoundError:
        app_path = REPO_ROOT / "person_yolo" / "web" / "app.py"
        spec = importlib.util.spec_from_file_location("person_yolo.web.app", app_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.main


main = _load_main()


if __name__ == "__main__":
    main()