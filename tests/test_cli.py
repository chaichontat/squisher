from typer.testing import CliRunner

from squisher import app


def test_cli_exposes_compress_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "compress" in result.stdout


def test_compress_accepts_czi_options(tmp_path) -> None:
    non_czi = tmp_path / "sample.txt"
    non_czi.write_text("not a czi")
    result = CliRunner().invoke(
        app,
        [
            "compress",
            str(non_czi),
            "--level",
            "90",
            "--tile-size",
            "512",
            "--tiff-maxworkers",
            "1",
            "--czi-tile-workers",
            "1",
            "--resume",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "Expected a .czi input file" in str(result.exception)
