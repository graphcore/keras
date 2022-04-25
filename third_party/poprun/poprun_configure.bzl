"""Configuration for distributed poprun tests"""

def _impl(repository_ctx):
    repository_ctx.file("BUILD", "")

    repository_ctx.template(
        "poprun_build_defs.bzl",
        Label("//third_party/poprun:poprun_build_defs.tpl"),
        {
            "{POPRUN_BINARY}": str(repository_ctx.which("poprun")),
            "{PYTHON_INTERPRETER}": str(repository_ctx.which("python")),
        },
    )

poprun_configure = repository_rule(
    implementation = _impl,
    local = True,
)
