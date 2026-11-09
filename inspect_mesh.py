"""Read one local mesh file and return numeric measurements.

Identical coordinates are welded. A gap stays open. Hole filling, normal
repair, and the library default merge distance are not used.
"""

from __future__ import annotations

import io
import math
import os
import stat
import struct
from pathlib import Path

NOTE = "A numeric check does not guarantee that the mesh looks right."
MAX_FILE_BYTES = 32 * 1024 * 1024
NAME_LIMIT = 80
_FORMATS = {"stl": "stl", "obj": "obj", "ply": "ply", "glb": "glb", "gltf": "gltf"}

_NEXT_EXPORT = "Export the mesh again as STL, OBJ, PLY, GLB, or glTF."
_NEXT_FORMAT = "Save the mesh as STL, OBJ, PLY, GLB, or glTF and pass that file."
_NEXT_EMPTY = "Export a mesh that contains triangles."
_NEXT_GLB = "Export a single GLB and pass that local file."
_NEXT_LOCAL = "Pass a local file path. From a file URI, drop the scheme first."
_NEXT_ENABLE = "Enable mesh-check so Hermes can prepare trimesh and pygltflib."
_NEXT_PATH = "Pass the path of a file that exists on this machine."
_NEXT_REGULAR = "Pass a regular file. Directories, devices, and pipes are skipped."
_NEXT_ARGUMENT = "Pass path as one non-empty string."
_NEXT_FINITE = "Remove NaN and infinite coordinates, then export the file again."
_NEXT_READ = "Choose a path the Hermes process can read."
_VOLUME_PLACES = 9


class CheckError(Exception):
    def __init__(self, error_type: str, error: str, next_step: str) -> None:
        self.error_type = error_type
        self.error = error
        self.next_step = next_step


def inspect_file(path, max_bytes=None):
    try:
        return _inspect(path, _cap(max_bytes))
    except CheckError as exc:
        return _fail(exc.error_type, exc.error, exc.next_step)
    except Exception as exc:
        return _fail(
            "internal_error",
            f"The check stopped with {type(exc).__name__}.",
            _NEXT_EXPORT,
        )


def _cap(max_bytes) -> int:
    if max_bytes is None:
        return MAX_FILE_BYTES
    try:
        requested = int(max_bytes)
    except (TypeError, ValueError):
        return MAX_FILE_BYTES
    if requested < 1:
        return MAX_FILE_BYTES
    if requested > MAX_FILE_BYTES:
        return MAX_FILE_BYTES
    return requested


def _fail(error_type: str, error: str, next_step: str) -> dict:
    return {
        "ok": False,
        "error_type": error_type,
        "error": error,
        "next_step": next_step,
        "note": NOTE,
    }


def _too_large(cap: int) -> CheckError:
    return CheckError(
        "too_large",
        f"The file is larger than {cap} bytes.",
        f"Export a smaller file. This check stops at {cap} bytes.",
    )


def _inspect(path, cap: int) -> dict:
    path = _intake_path(path)
    try:
        info = os.stat(path, follow_symlinks=True)
    except FileNotFoundError as exc:
        raise CheckError("not_found", "No file is at that path.", _NEXT_PATH) from exc
    except OSError as exc:
        raise CheckError("unreadable", "This process cannot open the path.", _NEXT_READ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise CheckError("not_a_file", "The path is not a regular file.", _NEXT_REGULAR)
    suffix = Path(path).suffix.lower().lstrip(".")
    fmt = _FORMATS.get(suffix)
    if fmt is None:
        raise CheckError(
            "unsupported_format",
            "The suffix is outside STL, OBJ, PLY, GLB, and glTF.",
            _NEXT_FORMAT,
        )
    if info.st_size == 0:
        raise CheckError("broken_file", "The file is empty.", _NEXT_EXPORT)
    if info.st_size > cap:
        raise _too_large(cap)
    try:
        with open(path, "rb") as handle:
            data = handle.read(cap + 1)
    except PermissionError as exc:
        raise CheckError("unreadable", "This process cannot read the file.", _NEXT_READ) from exc
    except OSError as exc:
        raise CheckError("unreadable", "This process cannot read the file.", _NEXT_READ) from exc
    if len(data) > cap:
        raise _too_large(cap)
    libs = _libraries()
    if libs is None:
        raise CheckError(
            "missing_dependency",
            "trimesh or numpy is not installed in this interpreter.",
            _NEXT_ENABLE,
        )
    np, trimesh = libs
    doc = None
    if fmt in {"glb", "gltf"}:
        doc = _gltf_document(data, fmt)
        _check_gltf_uris(path, doc, cap, len(data))
    try:
        loaded = _load_raw(fmt, data, path, trimesh)
    except CheckError:
        raise
    except Exception as exc:
        raise CheckError("broken_file", "The file could not be read as a mesh.", _NEXT_EXPORT) from exc
    triangles, other = _collect(loaded)
    if not triangles:
        if _known_empty(fmt, data, other, doc):
            raise CheckError("empty_mesh", "The file parsed and has no triangles.", _NEXT_EMPTY)
        raise CheckError("broken_file", "The file could not be read as a mesh.", _NEXT_EXPORT)
    measured = _measure(np, trimesh, triangles)
    return _result(path, fmt, data, doc, measured, other)


def _intake_path(path) -> str:
    """Classify one path string.

    The rows live in tests/test_mesh_check.py. An exact filesystem name
    that exists is kept. Trimming applies only when that exact string is
    not a file, so a real name that contains a shell comment is not rewritten.
    """
    if not isinstance(path, str) or "\x00" in path or path.strip() == "":
        raise CheckError("bad_argument", "path is missing or is not a string.", _NEXT_ARGUMENT)
    if _external_string(path):
        raise CheckError(
            "external_uri",
            "The path is a URI. This check reads a local file.",
            _NEXT_LOCAL,
        )
    if _stat_ok(path):
        return path
    trimmed = _shell_trim(path)
    if trimmed == "":
        raise CheckError("bad_argument", "path is missing or is not a string.", _NEXT_ARGUMENT)
    if trimmed != path and _external_string(trimmed):
        raise CheckError(
            "external_uri",
            "The path is a URI. This check reads a local file.",
            _NEXT_LOCAL,
        )
    return trimmed


def _stat_ok(path: str) -> bool:
    try:
        os.stat(path, follow_symlinks=True)
    except OSError:
        return False
    return True


def _shell_trim(path: str) -> str:
    current = path.strip()
    while True:
        if " #" in current:
            head, sep, _tail = current.rpartition(" #")
            if sep:
                current = head.strip()
                continue
        if current.endswith(";"):
            current = current.rstrip(";").strip()
            continue
        return current


def _external_string(path: str) -> bool:
    lowered = path.strip().lower()
    if "://" in path:
        return True
    return lowered.startswith(("http:", "https:", "file:", "ftp:"))


def _libraries():
    try:
        import numpy as np
        import trimesh
    except ImportError:
        return None
    return np, trimesh


def _gltf_document(data: bytes, fmt: str):
    try:
        from pygltflib import GLTF2
    except ImportError as exc:
        raise CheckError(
            "missing_dependency",
            "pygltflib is not installed in this interpreter.",
            _NEXT_ENABLE,
        ) from exc
    try:
        if fmt == "glb":
            return GLTF2.load_from_bytes(data)
        return GLTF2.gltf_from_json(data.decode("utf-8"))
    except Exception as exc:
        raise CheckError("broken_file", "The glTF file could not be parsed.", _NEXT_EXPORT) from exc


def _check_gltf_uris(path: str, doc, cap: int, used: int) -> None:
    parent = Path(os.path.abspath(path)).parent
    remaining = cap - used
    for uri in _uris(doc):
        kind = _uri_kind(uri)
        if kind == "embedded":
            continue
        if kind == "refuse":
            raise CheckError(
                "external_uri",
                "A glTF URI points outside this file's directory.",
                _NEXT_GLB,
            )
        candidate = Path(os.path.abspath(parent / uri))
        try:
            candidate.relative_to(parent)
        except ValueError as exc:
            raise CheckError(
                "external_uri",
                "A glTF URI points outside this file's directory.",
                _NEXT_GLB,
            ) from exc
        cursor = parent
        for part in candidate.relative_to(parent).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise CheckError(
                    "external_uri",
                    "A glTF URI uses a symlink. Export a single GLB.",
                    _NEXT_GLB,
                )
        if not candidate.exists():
            raise CheckError(
                "broken_file",
                "A glTF buffer file named by the URI is missing.",
                _NEXT_GLB,
            )
        try:
            info = os.lstat(candidate)
        except OSError as exc:
            raise CheckError("unreadable", "A glTF buffer could not be stated.", _NEXT_READ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise CheckError(
                "external_uri",
                "A glTF URI is not a regular file inside this file's directory.",
                _NEXT_GLB,
            )
        if info.st_size > remaining:
            raise _too_large(cap)
        remaining -= info.st_size


def _uris(doc) -> list[str]:
    found = []
    for bucket in ("buffers", "images"):
        for item in getattr(doc, bucket, None) or []:
            uri = getattr(item, "uri", None)
            if uri:
                found.append(str(uri))
    return found


def _uri_kind(uri: str) -> str:
    if uri.strip() == "" or any(ch in uri for ch in ("\x00", "\n", "\r")):
        return "refuse"
    lowered = uri.strip().lower()
    if lowered.startswith("data:"):
        return "embedded"
    if "://" in uri or lowered.startswith(("http:", "https:", "file:", "ftp:")):
        return "refuse"
    if uri.startswith(("/", "\\")) or "\\" in uri:
        return "refuse"
    if any(part == ".." for part in uri.split("/")):
        return "refuse"
    return "relative"


def _load_raw(fmt: str, data: bytes, path: str, trimesh):
    handle = io.BytesIO(data)
    if fmt == "stl":
        from trimesh.exchange.stl import load_stl

        return load_stl(handle)
    if fmt == "obj":
        from trimesh.exchange.obj import load_obj

        return load_obj(
            handle,
            split_objects=True,
            group_material=False,
            skip_materials=True,
            process=False,
        )
    if fmt == "ply":
        from trimesh.exchange.ply import load_ply

        return load_ply(handle, skip_materials=True, process=False)
    from trimesh.exchange.gltf import load_glb, load_gltf
    from trimesh.resolvers import FilePathResolver

    resolver = FilePathResolver(path, allow_anywhere=False)
    loader = load_glb if fmt == "glb" else load_gltf
    return loader(
        handle,
        resolver=resolver,
        skip_materials=True,
        merge_primitives=False,
        process=False,
    )


def _collect(loaded) -> tuple[list, int]:
    if not isinstance(loaded, dict):
        raise CheckError("broken_file", "The file could not be read as a mesh.", _NEXT_EXPORT)
    geometry = loaded.get("geometry") if isinstance(loaded.get("geometry"), dict) else None
    if geometry is not None:
        pairs = list(geometry.items())
    elif "vertices" in loaded:
        pairs = [("mesh", loaded)]
    else:
        pairs = []
    triangles = []
    other = 0
    for name, spec in pairs:
        vertices, faces = _spec_arrays(spec)
        if faces is None or _length(faces) == 0:
            if vertices is not None and _length(vertices) > 0:
                other += 1
            continue
        triangles.append((str(name), vertices, faces))
    return triangles, other


def _spec_arrays(spec):
    if isinstance(spec, dict):
        return spec.get("vertices"), spec.get("faces")
    return getattr(spec, "vertices", None), getattr(spec, "faces", None)


def _length(value) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _known_empty(fmt: str, data: bytes, other: int, doc) -> bool:
    if fmt in {"glb", "gltf"}:
        return doc is not None
    if fmt == "obj":
        return other > 0 or _obj_has_vertex(data)
    if fmt == "ply":
        return other > 0 or _ply_header(data)
    if fmt == "stl":
        return _stl_empty(data)
    return False


def _obj_has_vertex(data: bytes) -> bool:
    for line in data.decode("utf-8", "replace").splitlines():
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[0] == "v" and stripped[1].isspace():
            return True
    return False


def _ply_header(data: bytes) -> bool:
    return data.startswith(b"ply") and b"end_header" in data


def _stl_empty(data: bytes) -> bool:
    if len(data) == 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count == 0:
            return True
    text = data.decode("latin-1", "replace").lower()
    return "solid" in text and "endsolid" in text and "vertex" not in text


def _triangulate(np, faces):
    """Split each polygon into triangles from its first corner."""
    raw = np.asarray(faces)
    if raw.dtype == object or raw.ndim != 2:
        rows = []
        for face in faces:
            rows.extend(_fan(face))
        if not rows:
            raise CheckError("broken_file", "The mesh is not a list of triangles.", _NEXT_EXPORT)
        return np.asarray(rows, dtype=np.int64)
    width = int(raw.shape[1])
    if width < 3:
        raise CheckError("broken_file", "The mesh is not a list of triangles.", _NEXT_EXPORT)
    if width == 3:
        return raw.astype(np.int64, copy=False)
    rows = []
    for face in raw:
        rows.extend(_fan(face))
    return np.asarray(rows, dtype=np.int64)


def _fan(face) -> list[tuple[int, int, int]]:
    try:
        corners = [int(item) for item in face]
    except (TypeError, ValueError) as exc:
        raise CheckError("broken_file", "The mesh is not a list of triangles.", _NEXT_EXPORT) from exc
    if len(corners) < 3:
        raise CheckError("broken_file", "The mesh is not a list of triangles.", _NEXT_EXPORT)
    return [(corners[0], corners[index], corners[index + 1]) for index in range(1, len(corners) - 1)]


def _measure(np, trimesh, triangles: list) -> list[dict]:
    measured = []
    for name, vertices, faces in triangles:
        verts = np.asarray(vertices, dtype=np.float64)
        facs = _triangulate(np, faces)
        if verts.ndim != 2 or verts.shape[1] != 3 or facs.ndim != 2 or facs.shape[1] != 3:
            raise CheckError("broken_file", "The mesh is not a list of triangles.", _NEXT_EXPORT)
        if verts.shape[0] == 0 or facs.shape[0] == 0:
            continue
        if not np.isfinite(verts).all():
            raise CheckError(
                "non_finite",
                "A coordinate is NaN or infinite.",
                _NEXT_FINITE,
            )
        if int(facs.min()) < 0 or int(facs.max()) >= verts.shape[0]:
            raise CheckError("broken_file", "A face index is outside the vertex list.", _NEXT_EXPORT)
        stored = int(verts.shape[0])
        unique, inverse = np.unique(verts, axis=0, return_inverse=True)
        welded_faces = inverse[facs]
        try:
            mesh = trimesh.Trimesh(
                vertices=unique,
                faces=welded_faces,
                process=False,
                validate=False,
                use_embree=False,
            )
        except Exception as exc:
            raise CheckError("broken_file", "The mesh could not be measured.", _NEXT_EXPORT) from exc
        volume, volume_reason = _solid_volume(mesh)
        bounds = _bounds(unique)
        measured.append(
            {
                "name": _clip_name(name),
                "watertight": bool(mesh.is_watertight),
                "face_count": int(facs.shape[0]),
                "vertex_count": int(unique.shape[0]),
                "stored_vertex_count": stored,
                "volume": volume,
                "volume_reason": volume_reason,
                "bounds": bounds,
            }
        )
    if not measured:
        raise CheckError("empty_mesh", "The file parsed and has no triangles.", _NEXT_EMPTY)
    return measured


def _solid_volume(mesh):
    if not bool(mesh.is_watertight):
        return None, "not_closed"
    if not bool(mesh.is_winding_consistent):
        return None, "winding_inconsistent"
    number = float(mesh.volume)
    if not math.isfinite(number):
        return None, "volume_not_finite"
    if number <= 0.0:
        return None, "volume_not_positive"
    rounded = round(number, _VOLUME_PLACES)
    if rounded <= 0.0:
        return None, "volume_below_precision"
    return rounded, None


def _bounds(vertices) -> dict:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    mins = []
    maxs = []
    sizes = []
    for index in range(3):
        pub_min, pub_max, pub_size = _publish_span(float(low[index]), float(high[index]))
        mins.append(pub_min)
        maxs.append(pub_max)
        sizes.append(pub_size)
    return {"min": mins, "max": maxs, "size": sizes}


def _publish_scalar(value: float) -> float:
    """One coordinate at 9 decimal places.

    A nonzero value that those places turn into 0 stays in scientific form.
    """
    number = float(value)
    if not math.isfinite(number):
        raise CheckError("non_finite", "A coordinate is NaN or infinite.", _NEXT_FINITE)
    if number == 0.0:
        return 0.0
    rounded = round(number, _VOLUME_PLACES)
    if rounded != 0.0:
        return rounded
    return _scientific(number)


def _scientific(number: float) -> float:
    shown = float(format(float(number), ".6g"))
    if shown == 0.0:
        return float(number)
    return shown


def _publish_span(raw_min: float, raw_max: float) -> tuple[float, float, float]:
    """Publish one axis so size is the difference of the published ends.

    The difference is not rounded again. When both ends fall on one
    9-decimal value and the vertices still span a nonzero extent, each
    end keeps its offset from that value in scientific form.
    """
    low = float(raw_min)
    high = float(raw_max)
    pub_min = _publish_scalar(low)
    pub_max = _publish_scalar(high)
    if high == low:
        return pub_min, pub_max, 0.0
    delta = pub_max - pub_min
    if delta != 0.0:
        return pub_min, pub_max, delta
    anchor = pub_min
    shown_min = anchor if low == anchor else anchor + _scientific(low - anchor)
    shown_max = anchor if high == anchor else anchor + _scientific(high - anchor)
    if shown_min == shown_max:
        shown_min = low
        shown_max = high
    return shown_min, shown_max, shown_max - shown_min


def _clip_name(name: str) -> str:
    text = " ".join(str(name).split())
    if text == "":
        return "mesh"
    if len(text) > NAME_LIMIT:
        return text[:NAME_LIMIT]
    return text


def _result(path, fmt, data, doc, measured, other: int) -> dict:
    file_mesh_count, animation_count, animation_source, material_count, material_source = _counts(
        fmt, data, doc, len(measured)
    )
    scene_count, scene_source, node_count, node_source = _scene_counts(fmt, doc)
    disagree = file_mesh_count != len(measured) or other > 0
    bounds = _union_bounds(measured)
    face_count = sum(item["face_count"] for item in measured)
    vertex_count = sum(item["vertex_count"] for item in measured)
    stored_vertex_count = sum(item["stored_vertex_count"] for item in measured)
    if disagree or len(measured) != 1:
        if file_mesh_count != len(measured):
            reason = "mesh_count_disagrees"
        elif other > 0:
            reason = "unmeasured_geometry"
        else:
            reason = "not_one_mesh"
        watertight = None
        watertight_reason = reason
        volume = None
        volume_reason = reason
    else:
        only = measured[0]
        watertight = only["watertight"]
        watertight_reason = None
        volume = only["volume"]
        volume_reason = only["volume_reason"]
    body = {
        "ok": True,
        "note": NOTE,
        "path": path,
        "format": fmt,
        "units": "file",
        "watertight": watertight,
        "watertight_reason": watertight_reason,
        "volume": volume,
        "volume_reason": volume_reason,
        "bounds": bounds,
        "face_count": face_count,
        "vertex_count": vertex_count,
        "stored_vertex_count": stored_vertex_count,
        "vertices_welded": stored_vertex_count != vertex_count,
        "mesh_count": file_mesh_count,
        "measured_mesh_count": len(measured),
        "animation_count": animation_count,
        "animation_source": animation_source,
        "material_count": material_count,
        "material_source": material_source,
        "scene_count": scene_count,
        "scene_source": scene_source,
        "node_count": node_count,
        "node_source": node_source,
        "meshes": measured,
    }
    return body


def _counts(fmt: str, data: bytes, doc, measured_count: int):
    if fmt in {"glb", "gltf"} and doc is not None:
        return (
            _len_attr(doc, "meshes"),
            _len_attr(doc, "animations"),
            "gltf",
            _len_attr(doc, "materials"),
            "gltf",
        )
    if fmt == "obj":
        count, source = _obj_materials(data)
        return measured_count, 0, "not_in_format", count, source
    if fmt == "ply":
        count, source = _ply_materials(data)
        return measured_count, 0, "not_in_format", count, source
    return measured_count, 0, "not_in_format", 0, "not_in_format"


def _scene_counts(fmt: str, doc):
    if fmt in {"glb", "gltf"} and doc is not None:
        return _len_attr(doc, "scenes"), "gltf", _len_attr(doc, "nodes"), "gltf"
    return None, "not_in_format", None, "not_in_format"


def _len_attr(doc, name: str) -> int:
    value = getattr(doc, name, None)
    if value is None:
        return 0
    return len(value)


def _obj_materials(data: bytes):
    seen = []
    for line in data.decode("utf-8", "replace").splitlines():
        stripped = line.strip()
        if len(stripped) >= 7 and stripped[:6].lower() == "usemtl" and stripped[6].isspace():
            name = stripped[6:].strip()
            if name and name not in seen:
                seen.append(name)
    if seen:
        return len(seen), "usemtl"
    return 0, "not_in_format"


def _ply_materials(data: bytes):
    head, marker, _rest = data.partition(b"end_header")
    if not marker:
        return 0, "not_in_format"
    for line in head.decode("latin-1", "replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "material":
            try:
                return int(parts[2]), "ply_header"
            except ValueError:
                return 0, "ply_header"
    return 0, "not_in_format"


def _union_bounds(measured: list[dict]) -> dict:
    low = [min(item["bounds"]["min"][index] for item in measured) for index in range(3)]
    high = [max(item["bounds"]["max"][index] for item in measured) for index in range(3)]
    return {
        "min": low,
        "max": high,
        "size": [high[index] - low[index] for index in range(3)],
    }
