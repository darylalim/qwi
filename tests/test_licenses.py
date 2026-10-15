"""Code license files, per-model licenses, and the README table (PLAN.md §5.10)."""

import hashlib
import re
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of https://www.apache.org/licenses/LICENSE-2.0.txt
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def _models() -> dict:
    return tomllib.loads((ROOT / "models.toml").read_text())


def render_license_table(models: dict) -> str:
    """The README "Model licenses" table, generated from models.toml."""
    rows = ["| Role | Model | Revision | License |", "|---|---|---|---|"]
    for role, entry in models["roles"].items():
        lic = models["licenses"][entry["license"]]
        repo = entry["repo"]
        rows.append(
            f"| {role} | [`{repo}`](https://huggingface.co/{repo}) "
            f"| `{entry['revision'][:8]}` | [{lic['name']}]({lic['url']}) |"
        )
    return "\n".join(rows)


def test_pyproject_declares_apache_2() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    for name in project["license-files"]:
        assert (ROOT / name).is_file()


def test_license_is_canonical_apache_2() -> None:
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == APACHE_2_SHA256


def test_notice_disclaims_weights_and_research_license() -> None:
    notice = (ROOT / "NOTICE").read_text()
    assert "no model weights" in notice
    assert "Qwen Research License" in notice


def test_every_role_is_pinned_and_licensed() -> None:
    models = _models()
    assert models["roles"]
    for role, entry in models["roles"].items():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), role
        assert entry["license"] in models["licenses"], role


def test_readme_license_table_matches_models_toml() -> None:
    expected = render_license_table(_models())
    readme = (ROOT / "README.md").read_text()
    assert expected in readme, f"Update README → Model licenses to:\n\n{expected}\n"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_wheel_carries_license_files(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), "-q"], cwd=ROOT, check=True
    )
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for name in ("LICENSE", "NOTICE"):
        assert any(n.endswith(f".dist-info/licenses/{name}") for n in names), name
