"""Dependency-free compilation and auditable vendoring checks; no .pyc writes."""
import ast
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io import ROOT


def check():
    files = [p for p in ROOT.rglob("*.py") if not {".venv", "outputs", "__pycache__"}.intersection(p.relative_to(ROOT).parts)]
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    report = {"python_compile": "passed", "python_files": len(files)}
    pl_source = ROOT.parent / "PL-Stitch/models/vision_transformer.py"
    if pl_source.exists():
        source = pl_source.read_text(encoding="utf-8")
        expected = source.replace("from utils import trunc_normal_", "from torch.nn.init import trunc_normal_")
        expected = expected.replace("from timm.models.registry import register_model", "")
        local = (ROOT / "third_party/pl_stitch_vit.py").read_text(encoding="utf-8")
        if local.strip() != expected.strip():
            raise AssertionError("Unexpected PL-Stitch source changes")
        report["pl_stitch_two_import_changes_only"] = "passed"
    official = ROOT.parent / "imeanflow-main/imf.py"
    if official.exists():
        source = official.read_text(encoding="utf-8")
        source = source.replace('e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)', 'e = self.fixed["noise"]')
        source = source.replace('jax.random.uniform(self.make_rng("gen"), shape=(bz,))', 'self.fixed["drop_uniform"]')
        reference = (ROOT / "tests/reference/official_core.py").read_text(encoding="utf-8")
        def methods(text):
            return {n.name: n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.col_offset == 4}
        original, copied = methods(source), methods(reference)
        names = ["u_fn", "v_cond_fn", "v_fn", "cond_drop", "guidance_fn", "forward"]
        for name in names:
            if ast.dump(original[name], include_attributes=False) != ast.dump(copied[name], include_attributes=False):
                raise AssertionError("Reference method changed: " + name)
        report["official_reference_six_method_AST_parity"] = "passed_except_documented_fixed_rng_inputs"
    # Runtime source must contain no machine-specific Windows drive paths.
    for path in files:
        if re.search(r"\b[A-Za-z]:[\\/]", path.read_text(encoding="utf-8")):
            raise AssertionError("Machine path hardcoded: " + str(path))
    return report


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
