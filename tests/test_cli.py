from typer.testing import CliRunner

from squisher import app


def test_cli_exposes_compress_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "compress" in result.stdout


def test_compress_help_lists_czi_options() -> None:
    result = CliRunner().invoke(app, ["compress", "--help"])

    assert result.exit_code == 0
    assert "--level" in result.stdout
    assert "--tile-size" in result.stdout
    assert "--czi-tile-workers" in result.stdout
