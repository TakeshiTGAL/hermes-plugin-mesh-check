"""Numeric checks for local STL, OBJ, PLY, and glTF files."""

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

import inspect_mesh
import schemas
import tools
from make_mesh import (
    triangles,
    write_animated_glb,
    write_ascii_stl,
    write_gltf_external,
    write_indexed_obj,
    write_obj,
    write_ply,
    write_quad_obj,
    write_stl,
    write_two_primitive_glb,
)

NOTE = "A numeric check does not guarantee that the mesh looks right."
ROOT = Path(__file__).resolve().parents[1]


def _loads(raw):
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _check(path, **kwargs):
    return inspect_mesh.inspect_file(str(path), **kwargs)


def _cube(tmp_path):
    path = tmp_path / "cube.stl"
    write_stl(path, triangles())
    return path


def _snap(root: Path):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                digest = ""
            rows.append((str(path), info.st_mode, info.st_size, info.st_mtime_ns, info.st_nlink, digest))
    return rows


def test_cube_stl_is_one_closed_solid(tmp_path):
    body = _check(_cube(tmp_path))
    assert body["ok"] is True
    assert body["note"] == NOTE
    assert body["format"] == "stl"
    assert body["units"] == "file"
    assert body["watertight"] is True
    assert body["watertight_reason"] is None
    assert body["volume"] == pytest.approx(1.0)
    assert body["volume_reason"] is None
    assert body["face_count"] == 12
    assert body["vertex_count"] == 8
    assert body["stored_vertex_count"] == 36
    assert body["vertices_welded"] is True
    assert body["bounds"]["size"] == [1.0, 1.0, 1.0]
    assert body["mesh_count"] == 1
    assert body["animation_count"] == 0
    assert body["animation_source"] == "not_in_format"
    assert body["material_count"] == 0
    assert body["material_source"] == "not_in_format"
    assert body["scene_count"] is None
    assert body["scene_source"] == "not_in_format"
    assert body["node_count"] is None
    assert len(body["meshes"]) == 1
    assert body["meshes"][0]["volume"] == pytest.approx(1.0)


def test_obj_ply_and_ascii_stl_match_the_cube(tmp_path):
    obj = tmp_path / "cube.obj"
    ply = tmp_path / "cube.ply"
    ascii_stl = tmp_path / "cube_ascii.stl"
    write_indexed_obj(obj)
    write_ply(ply)
    write_ascii_stl(ascii_stl, [("cube", triangles())])
    for path in (obj, ply, ascii_stl):
        body = _check(path)
        assert body["ok"] is True, path.name
        assert body["watertight"] is True
        assert body["volume"] == pytest.approx(1.0)
        assert body["face_count"] == 12
        assert body["vertex_count"] == 8


def test_quad_obj_triangulates_to_a_closed_cube(tmp_path):
    path = tmp_path / "quad.obj"
    write_quad_obj(path)
    body = _check(path)
    assert body["ok"] is True
    assert body["face_count"] == 12
    assert body["watertight"] is True
    assert body["volume"] == pytest.approx(1.0)


def test_hole_does_not_report_the_surface_integral(tmp_path):
    path = tmp_path / "hole.stl"
    write_stl(path, triangles(drop_top=True))
    body = _check(path)
    assert body["ok"] is True
    assert body["watertight"] is False
    assert body["volume"] is None
    assert body["volume_reason"] == "not_closed"
    assert body["face_count"] == 10
    assert body["meshes"][0]["volume"] is None


def test_exact_duplicate_welds_and_hairline_stays_open(tmp_path):
    closed = tmp_path / "closed.stl"
    gap = tmp_path / "gap.stl"
    write_stl(closed, triangles())
    write_stl(gap, triangles(gap=1e-9))
    closed_body = _check(closed)
    gap_body = _check(gap)
    assert closed_body["watertight"] is True
    assert closed_body["vertex_count"] == 8
    assert gap_body["ok"] is True
    assert gap_body["watertight"] is False
    assert gap_body["vertex_count"] == 9
    assert gap_body["volume"] is None


def test_inverted_and_inconsistent_winding_omit_volume(tmp_path):
    inverted = tmp_path / "inverted.stl"
    mixed = tmp_path / "mixed.stl"
    write_stl(inverted, triangles(flip=True))
    write_stl(mixed, triangles(flip_first=True))
    inverted_body = _check(inverted)
    mixed_body = _check(mixed)
    assert inverted_body["watertight"] is True
    assert inverted_body["volume"] is None
    assert inverted_body["volume_reason"] == "volume_not_positive"
    assert mixed_body["volume"] is None
    assert mixed_body["meshes"][0]["volume"] is None


def _scale(tris, factor):
    return [[(x * factor, y * factor, z * factor) for x, y, z in tri] for tri in tris]


def test_rounded_nonpositive_volume_is_null(tmp_path):
    tiny = tmp_path / "tiny.stl"
    write_stl(tiny, _scale(triangles(), 1e-4))
    body = _check(tiny)
    assert body["ok"] is True
    assert body["watertight"] is True
    assert body["volume"] is None
    assert body["volume_reason"] == "volume_below_precision"
    assert body["meshes"][0]["volume"] is None
    assert body["meshes"][0]["volume_reason"] == "volume_below_precision"
    assert body["bounds"]["size"] == [0.0001, 0.0001, 0.0001]

    milli = tmp_path / "milli.stl"
    write_stl(milli, _scale(triangles(), 0.001))
    milli_body = _check(milli)
    assert milli_body["watertight"] is True
    assert milli_body["volume"] == pytest.approx(1e-9)
    assert milli_body["volume_reason"] is None

    dust = tmp_path / "dust.stl"
    write_stl(dust, _scale(triangles(), 1e-10))
    dust_body = _check(dust)
    assert dust_body["ok"] is True
    assert dust_body["watertight"] is True
    assert dust_body["vertex_count"] == 8
    assert dust_body["volume"] is None
    assert dust_body["volume_reason"] == "volume_below_precision"
    assert dust_body["bounds"]["size"][0] > 0
    assert dust_body["bounds"]["max"][0] > 0

    speck = tmp_path / "speck.stl"
    write_stl(speck, _scale(triangles(), 1e-16))
    speck_body = _check(speck)
    assert speck_body["ok"] is True
    assert speck_body["watertight"] is True
    assert speck_body["volume"] is None
    assert speck_body["volume_reason"] == "volume_below_precision"
    assert speck_body["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert speck_body["bounds"]["max"][0] > 0
    assert speck_body["bounds"]["size"][0] > 0
    assert speck_body["bounds"]["size"] == speck_body["meshes"][0]["bounds"]["size"]
    assert speck_body["volume"] != 0.0

    speck_obj = tmp_path / "speck.obj"
    write_obj(speck_obj, [("speck", _scale(triangles(), 1e-16))])
    speck_obj_body = _check(speck_obj)
    assert speck_obj_body["watertight"] is True
    assert speck_obj_body["volume"] is None
    assert speck_obj_body["volume_reason"] == "volume_below_precision"
    assert speck_obj_body["bounds"]["size"][0] > 0
    assert speck_obj_body["bounds"]["max"][0] > 0

    opened = tmp_path / "tiny_open.stl"
    write_stl(opened, _scale(triangles(drop_top=True), 1e-4))
    opened_body = _check(opened)
    assert opened_body["watertight"] is False
    assert opened_body["volume"] is None
    assert opened_body["volume_reason"] == "not_closed"

    pair = tmp_path / "big_and_tiny.stl"
    write_ascii_stl(
        pair,
        [("big", triangles()), ("tiny", _scale(triangles(), 1e-4))],
    )
    pair_body = _check(pair)
    assert pair_body["volume"] is None
    assert pair_body["volume_reason"] == "not_one_mesh"
    assert pair_body["meshes"][0]["volume"] == pytest.approx(1.0)
    assert pair_body["meshes"][1]["volume"] is None
    assert pair_body["meshes"][1]["volume_reason"] == "volume_below_precision"


def test_binary_shells_sum_inside_one_mesh(tmp_path):
    apart = tmp_path / "apart.stl"
    write_stl(apart, triangles() + triangles(shift=(3.0, 0.0, 0.0)))
    apart_body = _check(apart)
    assert apart_body["mesh_count"] == 1
    assert apart_body["watertight"] is True
    assert apart_body["volume"] == pytest.approx(2.0)
    assert apart_body["face_count"] == 24
    assert apart_body["vertex_count"] == 16

    overlap = tmp_path / "overlap.stl"
    write_stl(overlap, triangles() + triangles(shift=(0.5, 0.0, 0.0)))
    overlap_body = _check(overlap)
    assert overlap_body["mesh_count"] == 1
    assert overlap_body["watertight"] is True
    assert overlap_body["volume"] == pytest.approx(2.0)

    same = tmp_path / "same.stl"
    write_stl(same, triangles() + triangles())
    same_body = _check(same)
    assert same_body["mesh_count"] == 1
    assert same_body["watertight"] is False
    assert same_body["volume"] is None
    assert same_body["volume_reason"] == "not_closed"


def test_two_solids_are_not_one_volume(tmp_path):
    path = tmp_path / "two.stl"
    write_ascii_stl(
        path,
        [("left", triangles()), ("right", triangles(shift=(3.0, 0.0, 0.0)))],
    )
    body = _check(path)
    assert body["ok"] is True
    assert body["mesh_count"] == 2
    assert body["watertight"] is None
    assert body["watertight_reason"] == "not_one_mesh"
    assert body["volume"] is None
    assert body["volume_reason"] == "not_one_mesh"
    assert [item["volume"] for item in body["meshes"]] == pytest.approx([1.0, 1.0])
    assert body["bounds"]["size"][0] == pytest.approx(4.0)


def test_animated_glb_counts(tmp_path):
    path = tmp_path / "spin.glb"
    write_animated_glb(path)
    body = _check(path)
    assert body["ok"] is True
    assert body["format"] == "glb"
    assert body["watertight"] is True
    assert body["volume"] == pytest.approx(1.0)
    assert body["animation_count"] == 1
    assert body["animation_source"] == "gltf"
    assert body["material_count"] >= 1
    assert body["material_source"] == "gltf"
    assert body["mesh_count"] == 1
    assert body["scene_count"] == 1
    assert body["scene_source"] == "gltf"
    assert body["node_count"] >= 1
    assert body["node_source"] == "gltf"


def test_external_gltf_buffer_stays_in_the_directory(tmp_path):
    path = write_gltf_external(tmp_path)
    sibling = list(tmp_path.glob("*.bin"))
    assert sibling
    before = sibling[0].read_bytes()
    body = _check(path)
    assert body["ok"] is True
    assert body["format"] == "gltf"
    assert body["watertight"] is True
    assert body["volume"] == pytest.approx(1.0)
    assert sibling[0].read_bytes() == before


def test_broken_unsupported_too_large_and_empty(tmp_path):
    broken = tmp_path / "bad.stl"
    broken.write_bytes(b"this is not a mesh")
    broken_body = _check(broken)
    assert broken_body["ok"] is False
    assert broken_body["error_type"] == "broken_file"
    assert broken_body["next_step"] == "Export the mesh again as STL, OBJ, PLY, GLB, or glTF."
    assert broken_body["note"] == NOTE

    for name in ("model.fbx", "model.3mf", "model.txt", "model"):
        path = tmp_path / name
        path.write_bytes(b"abc")
        body = _check(path)
        assert body["error_type"] == "unsupported_format"
        assert body["next_step"] == "Save the mesh as STL, OBJ, PLY, GLB, or glTF and pass that file."

    huge = tmp_path / "huge.stl"
    fd = os.open(huge, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.lseek(fd, inspect_mesh.MAX_FILE_BYTES, os.SEEK_SET)
        os.write(fd, b"x")
    finally:
        os.close(fd)
    huge_body = _check(huge)
    assert huge_body["error_type"] == "too_large"
    assert huge_body["next_step"] == "Export a smaller file. This check stops at 33554432 bytes."

    small = tmp_path / "small.stl"
    small.write_bytes(b"0123456789abcdefEXTRA")
    small_body = _check(small, max_bytes=16)
    assert small_body["error_type"] == "too_large"
    assert "16" in small_body["next_step"]

    empty_file = tmp_path / "zero.stl"
    empty_file.write_bytes(b"")
    assert _check(empty_file)["error_type"] == "broken_file"

    empty_stl = tmp_path / "empty.stl"
    empty_stl.write_bytes(b"\0" * 80 + b"\0\0\0\0")
    empty_body = _check(empty_stl)
    assert empty_body["error_type"] == "empty_mesh"
    assert empty_body["next_step"] == "Export a mesh that contains triangles."

    points = tmp_path / "points.ply"
    write_ply(points, points=True)
    assert _check(points)["error_type"] == "empty_mesh"

    verts = tmp_path / "verts.obj"
    verts.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n", encoding="utf-8")
    assert _check(verts)["error_type"] == "empty_mesh"


def test_argument_path_and_format_edges(tmp_path):
    assert _check_raw(None)["error_type"] == "bad_argument"
    assert tools.handle_mesh_check(["cube.stl"])
    assert _loads(tools.handle_mesh_check(None))["error_type"] == "bad_argument"
    assert _loads(tools.handle_mesh_check({"path": ""}))["error_type"] == "bad_argument"
    assert _loads(tools.handle_mesh_check({"path": 12}))["error_type"] == "bad_argument"
    assert _check_raw("cube\0.stl")["error_type"] == "bad_argument"
    missing = _check(tmp_path / "missing.stl")
    assert missing["error_type"] == "not_found"
    assert missing["next_step"] == "Pass the path of a file that exists on this machine."
    assert _check(tmp_path)["error_type"] == "not_a_file"
    for uri in (
        "https://example.com/a.stl",
        "http://example.com/a.stl",
        "file:///tmp/a.stl",
        "HTTP://example.com/a.stl",
    ):
        body = inspect_mesh.inspect_file(uri)
        assert body["error_type"] == "external_uri"
        assert body["next_step"] == "Pass a local file path. From a file URI, drop the scheme first."
    cube = _cube(tmp_path)
    raw = tools.handle_mesh_check({"path": str(cube), "max_bytes": 1, "repair": True})
    body = _loads(raw)
    assert isinstance(raw, str)
    assert body["ok"] is True
    assert "repair" not in body
    raised = inspect_mesh.inspect_file(str(cube), max_bytes=10**18)
    assert raised["ok"] is True
    assert inspect_mesh.MAX_FILE_BYTES == 33554432


LOCAL_NEXT = "Pass a local file path. From a file URI, drop the scheme first."


def test_path_intake_matches_the_input_table(tmp_path, monkeypatch):
    """Every row of the path-intake table. Exact names that exist are not rewritten."""
    import socket

    def _blocked(*_args, **_kwargs):
        raise AssertionError("socket connect")

    monkeypatch.setattr(socket.socket, "connect", _blocked)

    plain = _cube(tmp_path)
    text = str(plain)
    blob = plain.read_bytes()

    def kind(raw):
        body = inspect_mesh.inspect_file(raw)
        dumped = json.dumps(body)
        assert "TOKENMARKER" not in dumped
        return body

    rows = [
        ("plain", text, "volume"),
        ("leading_space", " " + text, "volume"),
        ("trailing_space", text + " ", "volume"),
        ("both_spaces", " " + text + " ", "volume"),
        ("tabs", "\t" + text + "\t", "volume"),
        ("newline", text + "\n", "volume"),
        ("crlf", text + "\r\n", "volume"),
        ("ideographic_space", "\u3000" + text + "\u3000", "volume"),
        ("one_semicolon", text + ";", "volume"),
        ("two_semicolons", text + ";;", "volume"),
        ("semicolon_space", text + "; ", "volume"),
        ("comment", text + " # note", "volume"),
        ("hash_only", text + " #", "volume"),
        ("semicolon_and_comment", " " + text + " ; # note", "volume"),
        ("https", "https://example.com/a.stl", "uri"),
        ("http", "http://example.com/a.stl", "uri"),
        ("file_uri", "file://" + text, "uri"),
        ("file_uri_upper", "FILE://" + text, "uri"),
        ("ftp", "ftp://example.com/a.stl", "uri"),
        ("spaced_https", " https://example.com/a.stl ", "uri"),
        ("uri_with_comment", "file://" + text + " # note", "uri"),
        ("scheme_colon_slash", "notes://cabinet/a.stl", "uri"),
        ("empty", "", "bad_argument"),
        ("spaces", "   ", "bad_argument"),
        ("nul", "cube\0.stl", "bad_argument"),
        ("comment_only", " ; # note", "bad_argument"),
        ("missing", str(tmp_path / "missing.stl"), "not_found"),
    ]
    for name, raw, expect in rows:
        body = kind(raw)
        if expect == "volume":
            assert body["ok"] is True, name
            assert body["volume"] == pytest.approx(1.0), name
            assert body["path"] == text, name
            assert body["watertight"] is True, name
        elif expect == "uri":
            assert body["ok"] is False, name
            assert body["error_type"] == "external_uri", name
            assert body["next_step"] == LOCAL_NEXT, name
            assert body.get("volume") is None, name
        else:
            assert body["ok"] is False, name
            assert body["error_type"] == expect, name

    names = {
        "cube_\u65b9.stl": "ja",
        "\uff43\uff55\uff42\uff45.stl": "fullwidth",
        "Jos\u00e9.stl": "accent",
        "cube?.stl": "question",
        'cube".stl': "quote",
        "cube\\x.stl": "backslash",
        "CUBE.STL": "upper",
        ".cube.stl": "hidden",
    }
    for filename, label in names.items():
        path = tmp_path / filename
        path.write_bytes(blob)
        body = kind(str(path))
        assert body["ok"] is True, label
        assert body["volume"] == pytest.approx(1.0), label
        assert body["path"] == str(path), label

    jose = tmp_path / "Jose.stl"
    assert kind(str(jose))["error_type"] == "not_found"

    monkeypatch.chdir(tmp_path)
    relative = kind("cube.stl")
    assert relative["ok"] is True
    assert relative["volume"] == pytest.approx(1.0)
    assert kind("missing-cube.stl")["error_type"] == "not_found"

    assert kind(str(tmp_path))["error_type"] == "not_a_file"

    nest = tmp_path / "nest"
    nest.mkdir()
    closed = nest / "part.stl"
    write_stl(closed, triangles())
    weird = nest / "part.stl # kept.stl"
    write_stl(weird, triangles(drop_top=True))
    semi_name = nest / "part.stl;"
    semi_name.write_bytes(b"not-a-mesh")
    spaced = nest / " part.stl"
    write_stl(spaced, triangles(drop_top=True))

    exact_weird = kind(str(weird))
    assert exact_weird["ok"] is True
    assert exact_weird["watertight"] is False
    assert exact_weird["path"] == str(weird)
    typed_comment = kind(str(closed) + " # extra")
    assert typed_comment["volume"] == pytest.approx(1.0)
    assert typed_comment["path"] == str(closed)
    same_as_weird = kind(str(closed) + " # kept.stl")
    assert same_as_weird["watertight"] is False
    assert same_as_weird["path"] == str(weird)

    exact_semi = kind(str(semi_name))
    assert exact_semi["ok"] is False
    assert exact_semi["error_type"] == "unsupported_format"
    assert exact_semi.get("volume") is None
    typed_semi = kind(str(closed) + ";;")
    assert typed_semi["volume"] == pytest.approx(1.0)
    assert typed_semi["path"] == str(closed)

    exact_spaced = kind(str(spaced))
    assert exact_spaced["watertight"] is False
    assert exact_spaced["path"] == str(spaced)
    typed_space = kind(str(closed) + " ")
    assert typed_space["volume"] == pytest.approx(1.0)
    assert typed_space["path"] == str(closed)

    secret = tmp_path / ".git-credentials"
    secret.write_text("TOKENMARKER\n", encoding="utf-8")
    secret_body = kind(str(secret))
    assert secret_body["ok"] is False
    link = tmp_path / "secret-link.stl"
    link.symlink_to(secret)
    link_body = kind(str(link))
    assert link_body["ok"] is False
    assert link_body["error_type"] == "broken_file"


def _check_raw(path):
    return inspect_mesh.inspect_file(path)


def test_suffix_case_and_unicode_name(tmp_path):
    stl = _cube(tmp_path)
    upper = tmp_path / "CUBE.STL"
    upper.write_bytes(stl.read_bytes())
    assert _check(upper)["ok"] is True
    obj = tmp_path / "CUBE.OBJ"
    write_indexed_obj(obj)
    assert _check(obj)["format"] == "obj"
    ply = tmp_path / "CUBE.PLY"
    write_ply(ply)
    assert _check(ply)["format"] == "ply"
    named = tmp_path / "cube_\u03b1.stl"
    named.write_bytes(stl.read_bytes())
    body = _check(named)
    assert body["ok"] is True
    assert "cube_\u03b1.stl" in body["path"]
    raw = tools.handle_mesh_check({"path": str(named)})
    assert _loads(raw)["ok"] is True


def test_non_finite_coordinates(tmp_path):
    path = tmp_path / "bad.obj"
    path.write_text("o bad\nv nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    body = _check(path)
    assert body["ok"] is False
    assert body["error_type"] == "non_finite"
    assert body["next_step"] == "Remove NaN and infinite coordinates, then export the file again."
    assert body.get("volume") is None


def test_long_object_name_is_clipped(tmp_path):
    name = "N" * 200
    path = tmp_path / "long.obj"
    write_obj(path, [(name, triangles())])
    body = _check(path)
    assert body["ok"] is True
    assert len(body["meshes"][0]["name"]) <= 80
    assert name not in json.dumps(body)


def test_secret_header_and_proxy_env_stay_out(tmp_path, monkeypatch):
    path = tmp_path / "secret.stl"
    write_stl(path, triangles(), header=b"SECRETMARKER")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy-secret-value:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-secret-value:9")
    monkeypatch.setenv("ALL_PROXY", "http://proxy-secret-value:9")
    monkeypatch.setenv("http_proxy", "http://proxy-secret-value:9")
    monkeypatch.setenv("https_proxy", "http://proxy-secret-value:9")
    monkeypatch.setenv("all_proxy", "http://proxy-secret-value:9")
    monkeypatch.setenv("no_proxy", "proxy-secret-value")
    monkeypatch.setenv("NO_PROXY", "proxy-secret-value")
    monkeypatch.setenv("ftp_proxy", "http://proxy-secret-value:9")
    raw = json.dumps(_check(path), ensure_ascii=False)
    assert "SECRETMARKER" not in raw
    assert "proxy-secret-value" not in raw


def test_disk_fingerprint_symlink_and_hardlink(tmp_path):
    path = _cube(tmp_path)
    target = tmp_path / "hidden-target.stl"
    target.write_bytes(path.read_bytes())
    link = tmp_path / "link.stl"
    link.symlink_to(target)
    hard = tmp_path / "hard.stl"
    os.link(path, hard)
    before = _snap(tmp_path)
    body = _check(link)
    assert body["ok"] is True
    assert "hidden-target" not in json.dumps(body)
    assert _check(hard)["ok"] is True
    assert _check(tmp_path / "missing-later.stl")["ok"] is False
    assert _snap(tmp_path) == before


def test_fifo_is_not_opened(tmp_path):
    path = tmp_path / "pipe.stl"
    os.mkfifo(path)
    body = _check(path)
    assert body["error_type"] == "not_a_file"
    assert body["next_step"] == "Pass a regular file. Directories, devices, and pipes are skipped."


def test_obj_skips_mtl_and_counts_usemtl(tmp_path, monkeypatch):
    path = tmp_path / "painted.obj"
    write_obj(path, [("cube", triangles())], materials=("Red", "Blue", "Red"))
    canary = tmp_path / "sidecar.mtl"
    canary.write_text("CANARYMTLDATA\n", encoding="utf-8")
    opened = []
    real_open = open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy)
    body = _check(path)
    assert body["ok"] is True
    assert body["material_count"] == 2
    assert body["material_source"] == "usemtl"
    assert "Red" not in json.dumps({k: v for k, v in body.items() if k != "path"})
    assert not any(name.endswith(".mtl") for name in opened)
    assert "CANARYMTLDATA" not in json.dumps(body)

    remote = tmp_path / "remote.obj"
    remote.write_text(
        "mtllib http://127.0.0.1/evil.mtl\nusemtl Red\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    import socket

    def _blocked(*_args, **_kwargs):
        raise AssertionError("socket connect")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    remote_body = _check(remote)
    assert remote_body["ok"] is True
    assert remote_body["material_count"] >= 1


def test_gltf_uris_are_refused_before_use(tmp_path, monkeypatch):
    import socket

    def _blocked(*_args, **_kwargs):
        raise AssertionError("socket connect")

    monkeypatch.setattr(socket.socket, "connect", _blocked)

    def plant(name, uri):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": uri, "byteLength": 4}],
                    "meshes": [],
                }
            ),
            encoding="utf-8",
        )
        return path

    for name, uri in (
        ("remote.gltf", "https://example.com/a.bin"),
        ("fileuri.gltf", "file:///tmp/a.bin"),
        ("escape.gltf", "../outside.bin"),
        ("abs.gltf", "/tmp/a.bin"),
        ("slash.gltf", "subdir\\a.bin"),
    ):
        body = _check(plant(name, uri))
        assert body["error_type"] == "external_uri", name
        assert body["next_step"] == "Export a single GLB and pass that local file."

    missing = plant("missing.gltf", "missing.bin")
    missing_body = _check(missing)
    assert missing_body["error_type"] == "broken_file"
    assert "GLB" in missing_body["next_step"]

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"SECRETBUFDATA")
    link = tmp_path / "linked.bin"
    link.symlink_to(payload)
    linked = plant("linked.gltf", "linked.bin")
    linked_body = _check(linked)
    assert linked_body["error_type"] == "external_uri"
    assert "SECRETBUFDATA" not in json.dumps(linked_body)

    pipe = tmp_path / "pipe.bin"
    os.mkfifo(pipe)
    pipe_body = _check(plant("pipe.gltf", "pipe.bin"))
    assert pipe_body["error_type"] == "external_uri"


def test_missing_dependency(tmp_path, monkeypatch):
    path = _cube(tmp_path)
    monkeypatch.setattr(inspect_mesh, "_libraries", lambda: None)
    body = _check(path)
    assert body["error_type"] == "missing_dependency"
    assert body["next_step"] == "Enable mesh-check so Hermes can prepare trimesh and pygltflib."
    assert "pip install" not in body["next_step"]


def test_register_has_no_hook_and_handler_is_json():
    spec = importlib.util.spec_from_file_location(
        "mesh_check_plugin",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

    ctx = Ctx()
    module.register(ctx)
    assert ctx.hooks == []
    assert ctx.tools[0]["name"] == "mesh_check"
    assert ctx.tools[0]["toolset"] == "mesh-check"
    other = "echo hello-unchanged"
    assert other == "echo hello-unchanged"


def test_description_and_manifest_and_source_limits():
    description = schemas.MESH_CHECK["description"]
    text = " ".join(description.split())
    match = re.search(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)", text)
    sentence = text[: match.end()]
    assert len(sentence) <= 60
    assert not sentence.lower().startswith("read-only")
    assert "STL" in sentence and "watertight" in sentence
    assert "3D" in sentence and "OBJ" in sentence and "PLY" in sentence
    assert "GLB" in sentence and "glTF" in sentence
    assert NOTE in description
    assert len(description) <= 1024
    assert schemas.MESH_CHECK["parameters"]["additionalProperties"] is False
    manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert 'trimesh>=5.2.0,<6' in manifest
    assert 'pygltflib>=1.16.5,<2' in manifest
    assert 'requires_hermes: ">=0.21.4"' in manifest
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert NOTE in readme
    assert "hermes plugins install --no-enable --yes-deps" in readme
    assert "env -u HERMES_HOME" in readme
    assert "pip install" not in readme.lower()
    for path in ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in (
            "allow_remote=True",
            "urlopen",
            "subprocess",
            "import scipy",
            "os." + "environ",
        ):
            assert token not in source


def test_declared_packages_do_not_require_scipy():
    import importlib.metadata as meta

    for dist_name in ("trimesh", "pygltflib"):
        try:
            requires = meta.requires(dist_name) or []
        except meta.PackageNotFoundError:
            pytest.skip(f"{dist_name} is not installed")
        hard = [item for item in requires if "extra ==" not in item]
        for item in hard:
            name = item.split(";", 1)[0].strip().lower()
            assert not name.startswith("scipy")
            assert not name.startswith("networkx")
            assert not name.startswith("rtree")
            assert not name.startswith("shapely")
            assert not name.startswith("manifold3d")
            assert not name.startswith("httpx")


def test_tool_search_three_phrases():
    src = os.environ.get("HERMES_SRC", "")
    catalog_path = Path(src) / "tools" / "tool_search_catalog.py" if src else Path()
    if not catalog_path.is_file():
        pytest.skip("HERMES_SRC is not set")
    spec = importlib.util.spec_from_file_location("hermes_tool_search_catalog", catalog_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"tool search catalog is unavailable: {exc}")
    build_catalog = module.build_catalog
    search_catalog = module.search_catalog
    defs = [{"type": "function", "function": schemas.MESH_CHECK}]
    for index in range(8):
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": f"distractor_{index}",
                    "description": (
                        "Check the file before a task and count the box of an item in a list."
                    ),
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        )
    catalog = build_catalog(defs)
    phrases = (
        "is this STL watertight before a 3D print",
        "check the volume and bounding box of an OBJ",
        "count meshes animations and materials in a GLB before a game export",
    )
    for phrase in phrases:
        ranked = [item.name for item in search_catalog(catalog, phrase, limit=5)]
        assert "mesh_check" in ranked[:5], (phrase, ranked)


def _place(origin, side):
    ox, oy, oz = origin
    if isinstance(side, tuple):
        sx, sy, sz = side
    else:
        sx = sy = sz = side
    return [
        [(ox + x * sx, oy + y * sy, oz + z * sz) for x, y, z in tri]
        for tri in triangles()
    ]


def _write_double_ply(path, tris):
    points = [point for tri in tris for point in tri]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {len(tris)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for x, y, z in points:
        lines.append(f"{x} {y} {z}")
    for index in range(len(tris)):
        base = index * 3
        lines.append(f"3 {base} {base + 1} {base + 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_published_ends(body):
    bounds = body["bounds"]
    meshes = body["meshes"]
    for index in range(3):
        assert bounds["size"][index] == bounds["max"][index] - bounds["min"][index]
        assert bounds["min"][index] == min(item["bounds"]["min"][index] for item in meshes)
        assert bounds["max"][index] == max(item["bounds"]["max"][index] for item in meshes)
    for mesh in meshes:
        box = mesh["bounds"]
        for index in range(3):
            assert box["size"][index] == box["max"][index] - box["min"][index]
    if len(meshes) == 1:
        assert bounds == meshes[0]["bounds"]


def _public(path):
    body = _loads(tools.handle_mesh_check({"path": str(path)}))
    assert body["ok"] is True
    _assert_published_ends(body)
    return body


def test_published_span_matches_the_input_table(tmp_path):
    """Every row of the span table. size stays the difference of published ends."""
    numeric = (
        ("zero", 1.0, 1.0, (1.0, 1.0, 0.0)),
        ("unit", 0.0, 1.0, (0.0, 1.0, 1.0)),
        ("four", 0.0, 4e-10, (0.0, 4e-10, 4e-10)),
        ("five", 0.0, 5e-10, (0.0, 1e-9, 1e-9)),
        ("eight", 0.0, 8e-10, (0.0, 1e-9, 1e-9)),
        ("band", -4e-10, 4e-10, (-4e-10, 4e-10, 8e-10)),
        ("tenth", 0.0, 1e-4, (0.0, 0.0001, 0.0001)),
        ("neg_cube", -2.0, -1.0, (-2.0, -1.0, 1.0)),
    )
    for name, low, high, expected in numeric:
        published = inspect_mesh._publish_span(low, high)
        assert published == expected, name
        assert published[2] == published[1] - published[0], name
        assert inspect_mesh._publish_span(published[0], published[1]) == published, name

    restored = (
        ("offset", 1.0, 1.0 + 4e-10),
        ("center", 1.0 - 2e-10, 1.0 + 2e-10),
        ("band_at_1", 1.0 - 4e-10, 1.0 + 4e-10),
        ("neg_center", -1.0 - 2e-10, -1.0 + 2e-10),
        ("at_1000", 1000.0, 1000.0 + 4e-10),
        ("denorm", 0.0, 1e-320),
        ("half_up", 1.5e-9, 3e-9),
    )
    for name, low, high in restored:
        published = inspect_mesh._publish_span(low, high)
        assert published[2] == published[1] - published[0], name
        assert published[2] != 0.0, name
        assert inspect_mesh._publish_span(published[0], published[1]) == published, name
        if name != "half_up":
            assert round(published[2], 9) == 0.0 or name == "band_at_1", name
        if name == "band_at_1":
            assert published[2] != 1e-9, name

    band = _place((-4e-10, -4e-10, -4e-10), 8e-10)
    offset = _place((1.0, 1.0, 1.0), 4e-10)
    center = _place((1.0 - 2e-10, 1.0 - 2e-10, 1.0 - 2e-10), 4e-10)
    files = []

    unit = tmp_path / "unit.stl"
    write_stl(unit, triangles())
    files.append(("unit_stl", unit, lambda body: body["bounds"]["size"] == [1.0, 1.0, 1.0] and body["volume"] == pytest.approx(1.0)))

    tenth = tmp_path / "tenth.stl"
    write_stl(tenth, _place((0.0, 0.0, 0.0), 1e-4))
    files.append((
        "tenth_stl",
        tenth,
        lambda body: body["bounds"]["size"] == [0.0001, 0.0001, 0.0001] and body["volume"] is None and body["volume_reason"] == "volume_below_precision",
    ))

    band_obj = tmp_path / "band.obj"
    write_obj(band_obj, [("band", band)])
    files.append(("band_obj", band_obj, lambda body: body["bounds"] == {
        "min": [-4e-10, -4e-10, -4e-10],
        "max": [4e-10, 4e-10, 4e-10],
        "size": [8e-10, 8e-10, 8e-10],
    } and body["watertight"] is True and body["volume"] is None))

    band_stl = tmp_path / "band.stl"
    write_stl(band_stl, band)
    files.append(("band_stl", band_stl, lambda body: body["bounds"]["size"] == [8e-10, 8e-10, 8e-10] and body["bounds"]["min"] == [-4e-10, -4e-10, -4e-10]))

    off_obj = tmp_path / "off.obj"
    write_obj(off_obj, [("off", offset)])
    files.append(("offset_obj", off_obj, lambda body: _thin_at_one(body) and body["vertex_count"] == 8 and body["watertight"] is True))

    cen_obj = tmp_path / "cen.obj"
    write_obj(cen_obj, [("cen", center)])
    files.append(("offset_center_obj", cen_obj, lambda body: _thin_nonzero(body) and body["vertex_count"] == 8 and body["watertight"] is True))

    wide = tmp_path / "wide.obj"
    write_obj(wide, [("wide", _place((1.0 - 4e-10, 1.0 - 4e-10, 1.0 - 4e-10), 8e-10))])
    files.append(("band_at_1_obj", wide, lambda body: body["vertex_count"] == 8 and body["bounds"]["size"][0] != 0.0 and body["bounds"]["size"][0] != 1e-9 and body["volume"] is None))

    off_ascii = tmp_path / "off.ascii.stl"
    write_ascii_stl(off_ascii, [("off", offset)])
    files.append(("offset_ascii", off_ascii, lambda body: _thin_at_one(body) and body["vertex_count"] == 8))

    off_bin = tmp_path / "off.bin.stl"
    write_stl(off_bin, offset)
    files.append(("offset_binary_stl", off_bin, lambda body: body["vertex_count"] == 1 and body["bounds"]["size"] == [0.0, 0.0, 0.0] and body["volume"] is None))

    off_ply = tmp_path / "off.ply"
    write_ply(off_ply, offset)
    files.append(("offset_ply_float", off_ply, lambda body: body["vertex_count"] == 1 and body["bounds"]["size"] == [0.0, 0.0, 0.0] and body["volume"] is None))

    off_double = tmp_path / "off.double.ply"
    _write_double_ply(off_double, offset)
    files.append(("offset_ply_double", off_double, lambda body: _thin_at_one(body) and body["vertex_count"] == 8 and body["watertight"] is True))

    eight = tmp_path / "eight.obj"
    write_obj(eight, [("eight", _place((0.0, 0.0, 0.0), 8e-10))])
    files.append(("eight_from_zero", eight, lambda body: body["bounds"]["max"] == [1e-9, 1e-9, 1e-9] and body["bounds"]["size"] == [1e-9, 1e-9, 1e-9]))

    five = tmp_path / "five.obj"
    write_obj(five, [("five", _place((0.0, 0.0, 0.0), 5e-10))])
    files.append(("five_from_zero", five, lambda body: body["bounds"]["size"] == [1e-9, 1e-9, 1e-9] and body["bounds"]["max"] == [1e-9, 1e-9, 1e-9]))

    four = tmp_path / "four.obj"
    write_obj(four, [("four", _place((0.0, 0.0, 0.0), 4e-10))])
    files.append(("four_from_zero", four, lambda body: body["bounds"]["size"] == [4e-10, 4e-10, 4e-10] and body["bounds"]["max"] == [4e-10, 4e-10, 4e-10]))

    neg = tmp_path / "neg.obj"
    write_obj(neg, [("neg", _place((-2.0, -2.0, -2.0), 1.0))])
    files.append(("neg_cube_obj", neg, lambda body: body["bounds"]["min"] == [-2.0, -2.0, -2.0] and body["bounds"]["size"] == [1.0, 1.0, 1.0] and body["volume"] == pytest.approx(1.0)))

    neg_c = tmp_path / "negc.obj"
    write_obj(neg_c, [("negc", _place((-1.0 - 2e-10, -1.0 - 2e-10, -1.0 - 2e-10), 4e-10))])
    files.append(("neg_center_obj", neg_c, lambda body: _thin_nonzero(body) and body["vertex_count"] == 8))

    far = tmp_path / "far.obj"
    write_obj(far, [("far", _place((1000.0, 1000.0, 1000.0), 4e-10))])
    files.append(("at_1000_obj", far, lambda body: _thin_nonzero(body) and body["bounds"]["min"] == [1000.0, 1000.0, 1000.0]))

    speck = tmp_path / "speck.obj"
    write_obj(speck, [("speck", _place((0.0, 0.0, 0.0), 1e-16))])
    files.append(("speck_obj", speck, lambda body: body["bounds"]["min"] == [0.0, 0.0, 0.0] and body["bounds"]["size"][0] > 0 and body["volume"] is None))

    mixed = tmp_path / "mixed.obj"
    mixed_tris = [[(x, 1.0 + y * 4e-10, -4e-10 + z * 8e-10) for x, y, z in tri] for tri in triangles()]
    write_obj(mixed, [("mixed", mixed_tris)])
    files.append(("mixed_axes", mixed, lambda body: body["bounds"]["size"][0] == 1.0 and body["bounds"]["size"][2] == 8e-10 and _thin_component(body, 1)))

    union = tmp_path / "union.stl"
    write_ascii_stl(union, [
        ("left", _place((-4e-10, 0.0, 0.0), 4e-10)),
        ("right", _place((0.0, 0.0, 0.0), 4e-10)),
    ])
    files.append(("union_band", union, _union_band))

    for name, path, check in files:
        body = _public(path)
        assert check(body), name


def _thin_nonzero(body):
    size = body["bounds"]["size"]
    return body["volume"] is None and body["volume_reason"] == "volume_below_precision" and all(item != 0.0 and round(item, 9) == 0.0 for item in size)


def _thin_at_one(body):
    return _thin_nonzero(body) and body["bounds"]["min"] == [1.0, 1.0, 1.0] and all(item != 1.0 for item in body["bounds"]["max"])


def _thin_component(body, index):
    size = body["bounds"]["size"][index]
    return size != 0.0 and round(size, 9) == 0.0 and size == body["bounds"]["max"][index] - body["bounds"]["min"][index]


def _union_band(body):
    if body["mesh_count"] != 2 or body["volume"] is not None or body["volume_reason"] != "not_one_mesh":
        return False
    if body["bounds"]["size"] != [8e-10, 4e-10, 4e-10]:
        return False
    return all(item["bounds"]["size"] == [4e-10, 4e-10, 4e-10] and item["volume"] is None for item in body["meshes"])


def test_gltf_primitives_are_separate_measured_meshes(tmp_path):
    path = tmp_path / "two.glb"
    write_two_primitive_glb(path)
    body = _public(path)
    assert body["format"] == "glb"
    assert body["mesh_count"] == 1
    assert body["measured_mesh_count"] == 2
    assert body["watertight"] is None
    assert body["watertight_reason"] == "mesh_count_disagrees"
    assert body["volume"] is None
    assert body["volume_reason"] == "mesh_count_disagrees"
    assert body["face_count"] == 24
    assert len(body["meshes"]) == 2
    assert all(item["volume"] is not None and item["watertight"] is True for item in body["meshes"])
