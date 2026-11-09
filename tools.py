"""Handler for mesh_check. Returns one JSON string."""

import json


def _mesh():
    if __package__:
        from . import inspect_mesh
        return inspect_mesh
    import inspect_mesh
    return inspect_mesh


def handle_mesh_check(args, **_kwargs):
    inspect_mesh = _mesh()
    if isinstance(args, dict):
        body = inspect_mesh.inspect_file(args.get("path"))
    else:
        body = inspect_mesh.inspect_file(None)
    return json.dumps(body, allow_nan=False, ensure_ascii=False, sort_keys=True)
