"""Tool schema for one local 3D file check."""

MESH_CHECK = {
    "name": "mesh_check",
    "description": (
        "Inspect 3D STL, OBJ, PLY, GLB, or glTF watertightness. "
        "Check whether this STL is watertight before a 3D print. "
        "Check the volume and bounding box of an OBJ. "
        "Count meshes, animations, and materials in a GLB before a game export. "
        "A numeric check does not guarantee that the mesh looks right."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "description": "Local path of one STL, OBJ, PLY, GLB, or glTF file. Pass a filesystem path, not a URI.",
            },
        },
        "required": ["path"],
    },
}
