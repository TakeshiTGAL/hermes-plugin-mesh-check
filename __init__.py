"""Register the read-only mesh_check tool."""


def _load():
    # Hermes loads this file as a package. The flat import is only for a
    # loader that left __package__ empty.
    if __package__:
        from . import schemas, tools
        return schemas, tools
    import schemas
    import tools
    return schemas, tools


def register(ctx) -> None:
    schemas, tools = _load()
    schema = schemas.MESH_CHECK
    ctx.register_tool(
        name=schema["name"],
        toolset="mesh-check",
        schema=schema,
        handler=tools.handle_mesh_check,
        description=schema["description"],
    )
