"""Small mesh files built in the test process. Nothing is downloaded."""

import struct
from pathlib import Path

VERTS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
)
FACES = (
    (0, 3, 2),
    (0, 2, 1),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)


def triangles(shift=(0.0, 0.0, 0.0), gap=0.0, flip=False, flip_first=False, drop_top=False):
    rows = []
    for index, face in enumerate(FACES):
        if drop_top and index in {2, 3}:
            continue
        order = face
        if flip or (flip_first and index == 0):
            order = (face[0], face[2], face[1])
        tri = []
        for corner, vid in enumerate(order):
            x, y, z = VERTS[vid]
            if gap and index == 0 and corner == 0:
                x += gap
            tri.append((x + shift[0], y + shift[1], z + shift[2]))
        rows.append(tri)
    return rows


def write_stl(path: Path, tris, header=b"mesh-check") -> None:
    raw = header[:80].ljust(80, b"\0")
    records = []
    for tri in tris:
        flat = [coord for point in tri for coord in point]
        records.append(struct.pack("<12fH", 0.0, 0.0, 0.0, *flat, 0))
    path.write_bytes(raw + struct.pack("<I", len(records)) + b"".join(records))


def write_ascii_stl(path: Path, solids) -> None:
    lines = []
    for name, tris in solids:
        lines.append(f"solid {name}")
        for tri in tris:
            lines.append("  facet normal 0 0 0")
            lines.append("    outer loop")
            for x, y, z in tri:
                lines.append(f"      vertex {x} {y} {z}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_obj(path: Path, groups, materials=()) -> None:
    lines = []
    if materials:
        lines.append("mtllib sidecar.mtl")
        for name in materials:
            lines.append(f"usemtl {name}")
    index = 1
    for name, tris in groups:
        lines.append(f"o {name}")
        for tri in tris:
            for x, y, z in tri:
                lines.append(f"v {x} {y} {z}")
            lines.append(f"f {index} {index + 1} {index + 2}")
            index += 3
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_indexed_obj(path: Path) -> None:
    lines = ["o cube"]
    for x, y, z in VERTS:
        lines.append(f"v {x} {y} {z}")
    for face in FACES:
        a, b, c = face
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quad_obj(path: Path) -> None:
    lines = ["o cube"]
    for x, y, z in VERTS:
        lines.append(f"v {x} {y} {z}")
    quads = ((1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8))
    for quad in quads:
        lines.append("f " + " ".join(str(item) for item in quad))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ply(path: Path, tris=None, points=False, materials=0) -> None:
    if points:
        body = ["0 0 0", "1 0 0"]
        header = [
            "ply",
            "format ascii 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "element face 0",
            "property list uchar int vertex_indices",
        ]
        if materials:
            header.append(f"element material {materials}")
            header.append("property int dummy")
        header.append("end_header")
        path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
        return
    tris = tris if tris is not None else triangles()
    verts = [point for tri in tris for point in tri]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(tris)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for x, y, z in verts:
        lines.append(f"{x} {y} {z}")
    for index in range(len(tris)):
        base = index * 3
        lines.append(f"3 {base} {base + 1} {base + 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_animated_glb(path: Path) -> None:
    import numpy as np
    import trimesh
    from pygltflib import (
        GLTF2,
        Accessor,
        Animation,
        AnimationChannel,
        AnimationChannelTarget,
        AnimationSampler,
        BufferView,
        Material,
    )

    mesh = trimesh.Trimesh(
        vertices=np.array(VERTS, dtype=np.float64),
        faces=np.array(FACES, dtype=np.int64),
        process=False,
        validate=False,
    )
    mesh.export(path, file_type="glb")
    gltf = GLTF2.load(str(path))
    blob = bytearray(gltf.binary_blob() or b"")
    blob.extend(b"\x00" * ((4 - (len(blob) % 4)) % 4))
    time_offset = len(blob)
    blob.extend(struct.pack("<2f", 0.0, 1.0))
    quat_offset = len(blob)
    blob.extend(struct.pack("<8f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    gltf.set_binary_blob(bytes(blob))
    gltf.buffers[0].byteLength = len(blob)
    view_time = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=time_offset, byteLength=8))
    view_quat = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=quat_offset, byteLength=32))
    acc_time = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=view_time,
            componentType=5126,
            count=2,
            type="SCALAR",
            min=[0.0],
            max=[1.0],
        )
    )
    acc_quat = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(bufferView=view_quat, componentType=5126, count=2, type="VEC4")
    )
    gltf.animations.append(
        Animation(
            name="spin",
            samplers=[AnimationSampler(input=acc_time, output=acc_quat, interpolation="LINEAR")],
            channels=[
                AnimationChannel(
                    sampler=0,
                    target=AnimationChannelTarget(node=0, path="rotation"),
                )
            ],
        )
    )
    if not gltf.materials:
        gltf.materials.append(Material(name="clay"))
    gltf.save_binary(str(path))


def write_gltf_external(directory: Path) -> Path:
    import numpy as np
    import trimesh
    from pygltflib import GLTF2

    glb = directory / "source.glb"
    mesh = trimesh.Trimesh(
        vertices=np.array(VERTS, dtype=np.float64),
        faces=np.array(FACES, dtype=np.int64),
        process=False,
        validate=False,
    )
    mesh.export(glb, file_type="glb")
    doc = GLTF2.load(str(glb))
    out = directory / "cube.gltf"
    doc.save(str(out))
    glb.unlink()
    return out


def write_two_primitive_glb(path: Path) -> None:
    import numpy as np
    import trimesh
    from pygltflib import GLTF2

    mesh = trimesh.Trimesh(
        vertices=np.array(VERTS, dtype=np.float64),
        faces=np.array(FACES, dtype=np.int64),
        process=False,
        validate=False,
    )
    mesh.export(path, file_type="glb")
    doc = GLTF2.load(str(path))
    doc.meshes[0].primitives.append(doc.meshes[0].primitives[0])
    doc.save(str(path))
