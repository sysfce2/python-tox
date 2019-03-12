from __future__ import unicode_literals

import json
from collections import namedtuple

import pkg_resources
import six

from tox import reporter
from tox.config import DepConfig, get_py_project_toml
from tox.constants import BUILD_ISOLATED, BUILD_REQUIRE_SCRIPT

BuildInfo = namedtuple("BuildInfo", ["requires", "backend_module", "backend_object"])


def build(config, session):
    build_info = get_build_info(config.setupdir)
    package_venv = session.getvenv(config.isolated_build_env)
    package_venv.envconfig.deps_matches_subset = True

    # we allow user specified dependencies so the users can write extensions to
    # install additional type of dependencies (e.g. binary)
    user_specified_deps = package_venv.envconfig.deps
    package_venv.envconfig.deps = [DepConfig(r, None) for r in build_info.requires]
    package_venv.envconfig.deps.extend(user_specified_deps)

    if package_venv.setupenv():
        package_venv.finishvenv()

    build_requires = get_build_requires(build_info, package_venv, config.setupdir)
    # we need to filter out requirements already specified in pyproject.toml or user deps
    base_build_deps = {pkg_resources.Requirement(r.name).key for r in package_venv.envconfig.deps}
    build_requires_dep = [
        DepConfig(r, None)
        for r in build_requires
        if pkg_resources.Requirement(r).key not in base_build_deps
    ]
    if build_requires_dep:
        with package_venv.newaction("build_requires", package_venv.envconfig.envdir) as action:
            package_venv.run_install_command(packages=build_requires_dep, action=action)
        package_venv.finishvenv(package_venv)
    return perform_isolated_build(build_info, package_venv, config.distdir, config.setupdir)


def get_build_info(folder):
    toml_file = folder.join("pyproject.toml")

    # as per https://www.python.org/dev/peps/pep-0517/

    def abort(message):
        reporter.error("{} inside {}".format(message, toml_file))
        raise SystemExit(1)

    if not toml_file.exists():
        reporter.error("missing {}".format(toml_file))
        raise SystemExit(1)

    config_data = get_py_project_toml(toml_file)

    if "build-system" not in config_data:
        abort("build-system section missing")

    build_system = config_data["build-system"]

    if "requires" not in build_system:
        abort("missing requires key at build-system section")
    if "build-backend" not in build_system:
        abort("missing build-backend key at build-system section")

    requires = build_system["requires"]
    if not isinstance(requires, list) or not all(isinstance(i, six.text_type) for i in requires):
        abort("requires key at build-system section must be a list of string")

    backend = build_system["build-backend"]
    if not isinstance(backend, six.text_type):
        abort("build-backend key at build-system section must be a string")

    args = backend.split(":")
    module = args[0]
    obj = args[1] if len(args) > 1 else ""

    return BuildInfo(requires, module, obj)


def perform_isolated_build(build_info, package_venv, dist_dir, setup_dir):
    with package_venv.new_action(
        "perform-isolated-build", package_venv.envconfig.envdir
    ) as action:
        # need to start with an empty (but existing) source distribution folder
        if dist_dir.exists():
            dist_dir.remove(rec=1, ignore_errors=True)
        dist_dir.ensure_dir()

        result = package_venv._pcall(
            [
                package_venv.envconfig.envpython,
                BUILD_ISOLATED,
                str(dist_dir),
                build_info.backend_module,
                build_info.backend_object,
            ],
            returnout=True,
            action=action,
            cwd=setup_dir,
        )
        reporter.verbosity2(result)
        return dist_dir.join(result.split("\n")[-2])


def get_build_requires(build_info, package_venv, setup_dir):
    with package_venv.new_action("get-build-requires", package_venv.envconfig.envdir) as action:
        result = package_venv._pcall(
            [
                package_venv.envconfig.envpython,
                BUILD_REQUIRE_SCRIPT,
                build_info.backend_module,
                build_info.backend_object,
            ],
            returnout=True,
            action=action,
            cwd=setup_dir,
        )
        return json.loads(result.split("\n")[-2])