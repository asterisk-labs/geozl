"""Run the example notebooks against the installed geozl.

The install cell is skipped, so a notebook uses the geozl, and the Rumi, of the
environment, such as this checkout or a wheel about to be released. Each one runs
in its own temporary directory, so the data it downloads stays out of the tree.
A notebook whose Colab runtime is a GPU only runs with --gpu, on a CUDA machine.

    python tools/run_notebooks.py                   the CPU notebooks
    python tools/run_notebooks.py --gpu             the GPU notebooks
    python tools/run_notebooks.py --write zarr      store the outputs and render the pages

Any cell that raises fails the run. Outputs are only compared by the asserts in
the notebooks, since sizes and timings change from one machine to the next.
"""

import argparse
import json
import re
import sys
import tempfile

import nbformat
import render_notebooks
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS = render_notebooks.DOCS / "notebooks"
INSTALL = re.compile(r"^\s*[%!]pip install", re.M)


def is_gpu(raw):
    return raw["metadata"].get("accelerator") == "GPU"


def execute(path, timeout):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and INSTALL.search(cell.source):
            cell.source = ""
    with tempfile.TemporaryDirectory() as tmp:
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                resources={"metadata": {"path": tmp}})
        client.execute()
    return json.loads(nbformat.writes(nb))["cells"]


def store(path, text, raw, cells):
    """Copy outputs into the notebook and leave the rest of its JSON as it was."""
    for cell, done in zip(raw["cells"], cells, strict=True):
        if cell["cell_type"] != "code":
            continue
        cell["metadata"].pop("execution", None)
        if INSTALL.search("".join(cell["source"])):
            cell["outputs"], cell["execution_count"] = [], None
        else:
            cell["outputs"], cell["execution_count"] = done["outputs"], done["execution_count"]
    second = text.split("\n", 2)[1]
    indent = len(second) - len(second.lstrip())
    path.write_text(json.dumps(raw, indent=indent, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="notebooks to run, by file stem")
    parser.add_argument("--gpu", action="store_true", help="run the GPU notebooks instead")
    parser.add_argument("--write", action="store_true",
                        help="store the outputs in the notebooks and render the pages")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    args = parser.parse_args()

    failed = []
    for name, _, _ in render_notebooks.NOTEBOOKS:
        path = NOTEBOOKS / f"{name}.ipynb"
        text = path.read_text()
        raw = json.loads(text)
        if args.names and name not in args.names:
            continue
        if not args.names and is_gpu(raw) != args.gpu:
            continue
        print("run", path.relative_to(render_notebooks.DOCS.parent), flush=True)
        try:
            cells = execute(path, args.timeout)
        except CellExecutionError as e:
            print(e, file=sys.stderr)
            failed.append(name)
            continue
        if args.write:
            store(path, text, raw, cells)
    if failed:
        raise SystemExit("failed notebooks: " + ", ".join(failed))
    if args.write:
        render_notebooks.main()


if __name__ == "__main__":
    main()
