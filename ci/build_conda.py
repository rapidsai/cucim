#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run legacy Conda builds with the prepared candidate locks when present."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

CONTEXT_KEYS = (
    "RELEASE_CANDIDATE_TRAIN_SHA256",
    "RAPIDS_CANDIDATE_CONDA_CHANNEL",
    "RAPIDS_CANDIDATE_BUILD_LOCK_PACKAGE",
    "RAPIDS_CANDIDATE_HOST_LOCK_PACKAGE",
)


def indexed_channel(value: str) -> tuple[Path, dict]:
    """Accept only an existing absolute local channel with a noarch index."""
    path = Path(value)
    if (
        any(character in value for character in (",", "\n", "\r"))
        or not path.is_absolute()
        or not path.is_dir()
    ):
        raise ValueError("candidate Conda channels must be absolute local directories")
    path = path.resolve()
    if any(character in str(path) for character in (",", "\n", "\r")):
        raise ValueError("candidate Conda channel paths cannot contain separators")
    index = json.loads((path / "noarch" / "repodata.json").read_text())
    if not isinstance(index, dict):
        raise ValueError("candidate channel index must be an object")
    return path, index


def candidate_arguments(environment: dict[str, str], append_file: Path) -> list[str]:
    """Validate prepared context; partial context must never use ordinary mode."""
    values = [environment.get(key, "") for key in CONTEXT_KEYS]
    if not any(values):
        return []
    if not all(values):
        raise ValueError(
            "candidate Conda builds require complete train/channel/lock context"
        )
    train, channel, build_lock, host_lock = values
    if re.fullmatch(r"[0-9a-f]{64}", train) is None:
        raise ValueError("invalid candidate train digest")
    stem = rf"rapids-release-lock-{train[:12]}-linux-(?:64|aarch64)-cuda[0-9]+"
    if re.fullmatch(stem + "-build", build_lock) is None or host_lock != (
        build_lock.removesuffix("-build") + "-host"
    ):
        raise ValueError("candidate lock names must match the prepared train and scope")
    path, index = indexed_channel(channel)
    records = []
    for section in ("packages", "packages.conda"):
        packages = index.get(section, {})
        if not isinstance(packages, dict):
            raise ValueError("invalid candidate channel package index")
        records.extend(packages.values())
    names = {record.get("name") for record in records if isinstance(record, dict)}
    if not {build_lock, host_lock}.issubset(names):
        raise ValueError(
            "prepared candidate lock packages are missing from the local index"
        )
    append_file.write_text(
        json.dumps({"requirements": {"build": [build_lock], "host": [host_lock]}})
    )
    return [
        "--append-file",
        str(append_file),
        "--override-channels",
        "--channel",
        str(path),
        "--channel",
        "https://conda.anaconda.org/conda-forge",
        "--no-anaconda-upload",
    ]


def child_environment(
    command: list[str], candidate: bool, environment: dict[str, str]
) -> dict[str, str]:
    result = environment.copy()
    if candidate:
        # Conda-build resets its Conda context during solves. CLI channel
        # overrides alone are lost there; preserve the same allowlist in
        # the child environment, without changing production configuration.
        channels = [
            (
                Path(command[index + 1]).as_uri()
                if Path(command[index + 1]).is_absolute()
                else command[index + 1]
            )
            for index, value in enumerate(command)
            if value == "--channel"
        ]
        result["CONDA_CHANNELS"] = ",".join(channels)
    return result


def configuration_sources(environment: dict[str, str]) -> dict:
    executable = shutil.which("conda", path=environment.get("PATH"))
    if executable is None:
        raise ValueError("Conda executable is unavailable")
    result = subprocess.run(  # nosec B603
        [executable, "config", "--show-sources", "--json"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("cannot inspect Conda configuration sources")
    sources = json.loads(result.stdout)
    if not isinstance(sources, dict):
        raise ValueError("invalid Conda configuration source report")
    return sources


@contextmanager
def isolated_channels(command: list[str], environment: dict[str, str], temporary: Path):
    """Sanitize writable job-local channel configuration and restore exact bytes."""
    if environment.get("GITHUB_ACTIONS") != "true" or not Path("/.dockerenv").is_file():
        raise ValueError(
            "candidate legacy Conda isolation requires a disposable GitHub job container"
        )
    roots = []
    for key in ("RUNNER_TEMP", "CONDA_PREFIX"):
        path = Path(environment.get(key, ""))
        if not path.is_absolute() or not path.is_dir() or path.is_symlink():
            raise ValueError(f"candidate Conda isolation requires a local {key}")
        roots.append(path.resolve())
    forbidden = {
        key
        for key in environment
        if key.startswith("CONDA_")
        and "CHANNEL" in key
        and key != "CONDA_CHANNEL_PRIORITY"
    }
    if forbidden:
        raise ValueError(
            "unsupported inherited Conda channel environment configuration"
        )
    sources = configuration_sources(environment)
    saved = []
    unsupported = {
        "channel_alias",
        "custom_channels",
        "custom_multichannels",
        "migrated_channel_aliases",
        "migrated_custom_channels",
        "allowlist_channels",
        "denylist_channels",
        "whitelist_channels",
        "override_channels_enabled",
    }
    for name, settings in sources.items():
        if not isinstance(settings, dict) or unsupported.intersection(settings):
            raise ValueError(
                "unsupported Conda channel remapping or policy configuration"
            )
        if not {"channels", "default_channels"}.intersection(settings):
            continue
        if name in ("envvars", "cmd_line"):
            raise ValueError("unsupported non-file channel configuration")
        path = Path(name)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not any(path.is_relative_to(root) for root in roots)
        ):
            raise ValueError(
                "channel configuration source is outside the writable job scope"
            )
        if (
            path.resolve() != path
            or not stat.S_ISREG(path.stat().st_mode)
            or not path.stat().st_mode & 0o222
            or not os.access(path, os.W_OK)
        ):
            raise ValueError(
                "channel configuration source is not a regular writable job file"
            )
        original = path.read_bytes()
        try:
            document = yaml.safe_load(original)
        except yaml.YAMLError:
            raise ValueError("cannot parse job-local Conda configuration") from None
        if not isinstance(document, dict):
            raise ValueError("Conda configuration must be a YAML mapping")
        saved.append((path, original, stat.S_IMODE(path.stat().st_mode), document))
    child = child_environment(command, True, environment)
    configuration = temporary / "condarc.yaml"
    configuration.write_text(
        yaml.safe_dump(
            {
                "channels": child["CONDA_CHANNELS"].split(","),
                "default_channels": child["CONDA_CHANNELS"].split(","),
                "anaconda_upload": False,
            }
        )
    )
    child["CONDARC"] = str(configuration)
    try:
        for path, _, _, document in saved:
            sanitized = {
                key: value
                for key, value in document.items()
                if key not in ("channels", "default_channels")
            }
            path.write_text(yaml.safe_dump(sanitized))
        yield child
    finally:
        failed = False
        for path, original, mode, _ in saved:
            try:
                if path.is_symlink() or path.resolve() != path:
                    raise OSError("configuration target changed")
                path.write_bytes(original)
                path.chmod(mode)
            except OSError:
                failed = True
        if failed:
            raise OSError(
                "failed to restore one or more job-local Conda configuration files"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("conda/recipes/libcucim", "conda/recipes/cucim"),
        required=True,
    )
    parser.add_argument("--cpp-channel")
    parser.add_argument("--no-test", action="store_true")
    arguments = parser.parse_args()
    command = ["rapids-conda-retry", "build"]
    if arguments.no_test:
        command.append("--no-test")
    with tempfile.TemporaryDirectory(prefix="cucim-candidate-conda-") as temporary:
        try:
            candidate = candidate_arguments(os.environ, Path(temporary) / "append.json")
            if candidate and arguments.cpp_channel:
                indexed_channel(arguments.cpp_channel)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if arguments.cpp_channel:
            command.extend(("--channel", arguments.cpp_channel))
        command.extend(candidate)
        command.append(arguments.recipe)
        environment = dict(os.environ)
        if not candidate:
            return subprocess.run(command, env=environment, check=False).returncode  # nosec B603
        try:
            with isolated_channels(command, environment, Path(temporary)) as child:
                return subprocess.run(command, env=child, check=False).returncode  # nosec B603
        except (ValueError, OSError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
