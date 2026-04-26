import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader


def parse_args():
    parser = argparse.ArgumentParser(description="Render canonical CHNS VTK output into PNG/PDF frames.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--field", default="phase")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--figsize", nargs=2, type=float, default=(3.0, 9.0), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_field(vtu_path, field_name):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    grid = reader.GetOutput()

    points = vtk_to_numpy(grid.GetPoints().GetData())[:, :2]
    values = vtk_to_numpy(grid.GetPointData().GetArray(field_name))

    rounded_points = np.round(points, decimals=12)
    _, unique_idx = np.unique(rounded_points, axis=0, return_index=True)
    unique_idx.sort()
    rounded_points = rounded_points[unique_idx]
    values = values[unique_idx]

    xs = np.unique(rounded_points[:, 0])
    ys = np.unique(rounded_points[:, 1])
    field = np.full((len(ys), len(xs)), np.nan)
    x_index = {value: idx for idx, value in enumerate(xs)}
    y_index = {value: idx for idx, value in enumerate(ys)}

    for (x_coord, y_coord), value in zip(rounded_points, values):
        field[y_index[y_coord], x_index[x_coord]] = value

    return xs, ys, field


def render_frame(vtu_path, field_name, output_base, vmin, vmax, figsize, dpi):
    xs, ys, field = load_field(vtu_path, field_name)
    levels = np.linspace(vmin, vmax, 200)

    fig, axes = plt.subplots(figsize=figsize)
    axes.contourf(xs, ys, field, levels=levels, extend="both")
    axes.set_aspect("equal")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 3.0)
    axes.set_xticks([])
    axes.set_yticks([])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f"{output_base}.png", dpi=dpi)
    fig.savefig(f"{output_base}.pdf", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir or f"output/chns-{args.benchmark}")
    output_prefix = args.output_prefix or f"output/chns-{args.benchmark}"

    frames = sorted(input_dir.glob("*.vtu"))
    if not frames:
        raise SystemExit(f"No VTU frames found in {input_dir}")

    for index, vtu_path in enumerate(frames):
        output_base = f"{output_prefix}-t{index:04d}"
        render_frame(vtu_path, args.field, output_base, args.vmin, args.vmax, tuple(args.figsize), args.dpi)
        print(f"Rendered {output_base}")


if __name__ == "__main__":
    main()
