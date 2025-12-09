#
# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from click.testing import CliRunner

from cucim.clara.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
