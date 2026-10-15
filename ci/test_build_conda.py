# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate legacy Conda boundary tests, including real Conda-build solves."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess  # nosec B404
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).with_name("build_conda.py")
SPEC = importlib.util.spec_from_file_location("candidate_conda", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TRAIN = "a" * 64
BUILD_LOCK = f"rapids-release-lock-{TRAIN[:12]}-linux-64-cuda12-build"
HOST_LOCK = BUILD_LOCK.removesuffix("build") + "host"


def context(channel):
    return dict(
        zip(
            MODULE.CONTEXT_KEYS,
            (TRAIN, str(channel), BUILD_LOCK, HOST_LOCK),
            strict=True,
        )
    )


def index(channel, records):
    (channel / "noarch").mkdir(parents=True, exist_ok=True)
    (channel / "noarch" / "repodata.json").write_text(
        json.dumps({"packages": records, "packages.conda": {}, "repodata_version": 1})
    )


@pytest.fixture
def channel(tmp_path):
    path = tmp_path / "candidate"
    index(
        path,
        {f"{name}-0-0.tar.bz2": {"name": name} for name in (BUILD_LOCK, HOST_LOCK)},
    )
    return path


def test_append_keeps_build_and_host_locks_separate(channel, tmp_path):
    append = tmp_path / "append.json"
    arguments = MODULE.candidate_arguments(context(channel), append)
    assert json.loads(append.read_text()) == {  # nosec B101
        "requirements": {"build": [BUILD_LOCK], "host": [HOST_LOCK]}
    }
    assert arguments == [  # nosec B101
        "--append-file",
        str(append),
        "--override-channels",
        "--channel",
        str(channel),
        "--channel",
        "https://conda.anaconda.org/conda-forge",
        "--no-anaconda-upload",
    ]


@pytest.mark.parametrize("missing", MODULE.CONTEXT_KEYS)
def test_partial_candidate_context_rejected(channel, tmp_path, missing):
    environment = context(channel)
    environment.pop(missing)
    with pytest.raises(ValueError, match="complete"):
        MODULE.candidate_arguments(environment, tmp_path / "append.json")


@pytest.mark.parametrize(
    "replacement",
    ("wrong", "rapids-release-lock-" + "b" * 12 + "-linux-64-cuda12-build"),
)
def test_wrong_lock_scope_rejected(channel, tmp_path, replacement):
    environment = context(channel)
    environment["RAPIDS_CANDIDATE_BUILD_LOCK_PACKAGE"] = replacement
    with pytest.raises(ValueError, match="lock names"):
        MODULE.candidate_arguments(environment, tmp_path / "append.json")


@pytest.mark.parametrize(
    "value", ("conda-forge", "https://example.invalid/channel", "file:///tmp/channel")
)
def test_nonlocal_channel_rejected(value):
    with pytest.raises(ValueError, match="absolute local"):
        MODULE.indexed_channel(value)


def test_missing_lock_rejected(channel, tmp_path):
    index(channel, {})
    with pytest.raises(ValueError, match="missing"):
        MODULE.candidate_arguments(context(channel), tmp_path / "append.json")


@pytest.mark.parametrize("python_build", (False, True))
def test_ordinary_argv_unchanged(tmp_path, monkeypatch, python_build):
    for key in MODULE.CONTEXT_KEYS:
        monkeypatch.delenv(key, raising=False)
    arguments = ["--recipe", "conda/recipes/libcucim"]
    expected = ["rapids-conda-retry", "build", "conda/recipes/libcucim"]
    if python_build:
        arguments = [
            "--no-test",
            "--cpp-channel",
            str(tmp_path),
            "--recipe",
            "conda/recipes/cucim",
        ]
        expected = [
            "rapids-conda-retry",
            "build",
            "--no-test",
            "--channel",
            str(tmp_path),
            "conda/recipes/cucim",
        ]
    calls = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append(argv) or subprocess.CompletedProcess(argv, 0)
        ),
    )
    assert MODULE.main() == 0  # nosec B101
    assert calls == [expected]  # nosec B101


def test_candidate_cpp_must_be_indexed_before_execution(channel, tmp_path, monkeypatch):
    for key, value in context(channel).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--recipe",
            "conda/recipes/cucim",
            "--cpp-channel",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not execute build"),
    )
    with pytest.raises(SystemExit) as error:
        MODULE.main()
    assert error.value.code == 2  # nosec B101


def package(channel, name, version, *, constrains=()):
    record = {
        "name": name,
        "version": version,
        "build": "0",
        "build_number": 0,
        "subdir": "noarch",
        "noarch": "generic",
        "depends": [],
        "constrains": list(constrains),
    }
    directory = channel / "noarch"
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(directory / f"{name}-{version}-0.tar.bz2", "w:bz2") as archive:
        for path, content in {
            "info/index.json": json.dumps(record).encode(),
            "info/files": b"",
        }.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


@pytest.mark.parametrize(
    "build_requirement,host_requirement,succeeds",
    (
        ("widget", "widget", True),
        ("widget ==2", "widget", False),
        ("widget", "widget ==1", False),
    ),
)
def test_real_conda_build_append_finalize_and_channel_override(
    tmp_path, monkeypatch, build_requirement, host_requirement, succeeds
):
    if not Path("/.dockerenv").is_file():
        pytest.skip(
            "real exclusive-channel solve requires a disposable Linux container, never host config mutation"
        )
    pytest.importorskip("conda_build")
    from conda.models.channel import Channel
    from conda_index.api import update_index

    channel = tmp_path / "channel"
    for version in ("1", "2"):
        package(channel, "widget", version)
    for lock, version in ((BUILD_LOCK, "1"), (HOST_LOCK, "2")):
        package(channel, lock, "0", constrains=(f"widget =={version}",))
    update_index(channel)
    append = tmp_path / "append.json"
    MODULE.candidate_arguments(context(channel), append)
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "meta.yaml").write_text(
        "package:\n  name: boundary-fixture\n  version: '1'\nbuild:\n  number: 0\n"
        f"requirements:\n  build:\n    - {build_requirement}\n  host:\n    - {host_requirement}\n"
    )
    configuration = tmp_path / "condarc"
    configuration.write_text(
        "channels:\n  - https://example.invalid/must-not-be-contacted\n"
        "  - defaults\ndefault_channels:\n  - https://example.invalid/defaults\n"
        "anaconda_upload: true\n"
    )
    environment = {
        **os.environ,
        "CONDARC": str(configuration),
        "CONDA_BLD_PATH": str(tmp_path / "build"),
        "CONDA_PKGS_DIRS": str(tmp_path / "pkgs"),
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "CONDA_OFFLINE": "true",
        "GITHUB_ACTIONS": "true",
        "RUNNER_TEMP": str(tmp_path),
        "CONDA_PREFIX": str(Path(sys.executable).parent.parent),
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--file",
        str(tmp_path / "rendered.yaml"),
        "--no-source",
        "--append-file",
        str(append),
        "--override-channels",
        "--channel",
        str(channel),
        str(recipe),
    ]
    original = configuration.read_bytes()
    with MODULE.isolated_channels(command, environment, tmp_path) as child:
        shown = subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "conda",
                "config",
                "--show",
                "channels",
                "default_channels",
                "--json",
            ],
            env=child,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        settings = json.loads(shown.stdout)
        for record in settings["default_channels"]:
            assert record["auth"] is None and record["token"] is None  # nosec B101
        settings["default_channels"] = [
            Channel(**record).base_url for record in settings["default_channels"]
        ]
        assert settings == {
            "channels": [channel.as_uri()],
            "default_channels": [channel.as_uri()],
        }  # nosec B101
        result = subprocess.run(  # nosec B603
            command,
            env=child,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert configuration.read_bytes() == original  # nosec B101
    if succeeds:
        assert result.returncode == 0, result.stderr  # nosec B101
        rendered = yaml.safe_load((tmp_path / "rendered.yaml").read_text())
        for section, lock, version in (
            ("build", BUILD_LOCK, "1"),
            ("host", HOST_LOCK, "2"),
        ):
            dependencies = rendered["requirements"][section]
            assert any(value.startswith(f"widget {version} ") for value in dependencies)  # nosec B101
            assert any(value.startswith(lock + " 0 ") for value in dependencies)  # nosec B101
    else:
        assert result.returncode != 0  # nosec B101
        assert "widget" in result.stderr  # nosec B101
    assert "example.invalid" not in result.stdout + result.stderr  # nosec B101


@pytest.mark.parametrize("failure", (False, True))
def test_configuration_bytes_modes_and_unrelated_settings_restored(
    tmp_path, monkeypatch, failure
):
    configuration = tmp_path / "condarc"
    original = b"# preserve comments\nchannels: [rapidsai-nightly, conda-forge]\nssl_verify: false\n"
    configuration.write_bytes(original)
    configuration.chmod(0o640)
    monkeypatch.setattr(
        MODULE,
        "configuration_sources",
        lambda env: {str(configuration): yaml.safe_load(original)},
    )
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/.dockerenv" or original_is_file(path),
    )
    environment = {
        "GITHUB_ACTIONS": "true",
        "RUNNER_TEMP": str(tmp_path),
        "CONDA_PREFIX": str(tmp_path),
        "CONDA_PY": "312",
        "CONDA_OVERRIDE_CUDA": "12.9",
    }

    def run():
        with MODULE.isolated_channels(
            ["build", "--channel", str(tmp_path)], environment, tmp_path
        ) as child:
            assert yaml.safe_load(configuration.read_bytes()) == {"ssl_verify": False}  # nosec B101
            assert child["CONDA_CHANNELS"] == tmp_path.as_uri()  # nosec B101
            assert child["CONDA_PY"] == "312"  # nosec B101
            assert "CONDA_CHANNELS" not in environment  # nosec B101
            if failure:
                raise RuntimeError("simulated failed build")

    if failure:
        with pytest.raises(RuntimeError, match="simulated"):
            run()
    else:
        run()
    assert configuration.read_bytes() == original  # nosec B101
    assert configuration.stat().st_mode & 0o777 == 0o640  # nosec B101


def test_noncontainer_refuses_configuration_mutation(tmp_path):
    with pytest.raises(ValueError, match="disposable"):
        with MODULE.isolated_channels([], {}, tmp_path):
            pytest.fail("must not start")


def test_candidate_disables_inherited_upload_setting(channel, tmp_path, monkeypatch):
    pytest.importorskip("conda_build")
    from conda.base.context import context as conda_context
    from conda_build.cli.main_build import parse_args

    monkeypatch.setenv("CONDA_ANACONDA_UPLOAD", "true")
    conda_context.__init__()
    try:
        assert conda_context.anaconda_upload is True  # nosec B101
        arguments = MODULE.candidate_arguments(
            context(channel), tmp_path / "append.json"
        )
        _, parsed = parse_args([*arguments, "conda/recipes/cucim"])
        assert parsed.anaconda_upload is False  # nosec B101
    finally:
        monkeypatch.delenv("CONDA_ANACONDA_UPLOAD")
        conda_context.__init__()


@pytest.fixture
def job_configuration(tmp_path, monkeypatch):
    configuration = tmp_path / "source.yaml"
    original = b"channels: [rapidsai-nightly]\nssl_verify: true\n"
    configuration.write_bytes(original)
    environment = {
        "GITHUB_ACTIONS": "true",
        "RUNNER_TEMP": str(tmp_path),
        "CONDA_PREFIX": str(tmp_path),
    }
    monkeypatch.setattr(
        MODULE,
        "configuration_sources",
        lambda env: {str(configuration): yaml.safe_load(original)},
    )
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == "/.dockerenv" or original_is_file(path),
    )
    return configuration, original, environment


@pytest.mark.parametrize(
    "problem", ("symlink", "readonly", "outside", "environment", "remapping")
)
def test_unsafe_configuration_refused_before_writes(
    job_configuration, tmp_path, monkeypatch, problem
):
    configuration, original, environment = job_configuration
    if problem == "symlink":
        target = tmp_path / "real.yaml"
        configuration.rename(target)
        configuration.symlink_to(target)
    elif problem == "readonly":
        configuration.chmod(0o400)
    elif problem == "outside":
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        environment.update(RUNNER_TEMP=str(allowed), CONDA_PREFIX=str(allowed))
    elif problem == "environment":
        environment["CONDA_CHANNELS"] = "rapidsai-nightly"
    else:
        monkeypatch.setattr(
            MODULE,
            "configuration_sources",
            lambda env: {
                str(configuration): {
                    "custom_channels": {"rapidsai-nightly": "https://example.invalid"}
                }
            },
        )
    with pytest.raises(ValueError, match="configuration|scope|writable"):
        with MODULE.isolated_channels(
            ["build", "--channel", str(tmp_path)], environment, tmp_path
        ):
            pytest.fail("must not start")
    assert configuration.read_bytes() == original  # nosec B101


def test_partial_sanitation_failure_restores_all_files(
    job_configuration, tmp_path, monkeypatch
):
    first, original, environment = job_configuration
    second = tmp_path / "second.yaml"
    second.write_bytes(original)
    monkeypatch.setattr(
        MODULE,
        "configuration_sources",
        lambda env: {str(path): yaml.safe_load(original) for path in (first, second)},
    )
    original_write = Path.write_text

    def write(path, data, *args, **kwargs):
        if path == second:
            raise OSError("simulated sanitation failure")
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match="simulated"):
        with MODULE.isolated_channels(
            ["build", "--channel", str(tmp_path)], environment, tmp_path
        ):
            pytest.fail("must not start")
    assert first.read_bytes() == original  # nosec B101
    assert second.read_bytes() == original  # nosec B101


def test_nonchannel_configuration_source_is_untouched(
    job_configuration, tmp_path, monkeypatch
):
    configuration, original, environment = job_configuration
    monkeypatch.setattr(
        MODULE,
        "configuration_sources",
        lambda env: {"/unknown/read-only/condarc": {"ssl_verify": True}},
    )
    with MODULE.isolated_channels(
        ["build", "--channel", str(tmp_path)], environment, tmp_path
    ):
        assert configuration.read_bytes() == original  # nosec B101


if __name__ == "__main__":
    # Exercise the real API with strict finalization: conda-render's CLI instead
    # permits unsatisfiable variants and can exit zero without producing a lock.
    from conda.base.context import context as conda_context
    from conda_build import api
    from conda_build.cli.main_render import parse_args
    from conda_build.config import get_channel_urls, get_or_merge_config

    _, parsed = parse_args(sys.argv[1:])
    conda_context.__init__(argparse_args=parsed)
    config = get_or_merge_config(None, **vars(parsed))
    config.channel_urls = get_channel_urls(vars(parsed))
    result = api.render(
        parsed.recipe, config=config, permit_unsatisfiable_variants=False
    )
    api.output_yaml(result[0][0], parsed.file)
