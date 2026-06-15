[CmdletBinding()]
param(
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runningOnWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [System.Runtime.InteropServices.OSPlatform]::Windows
)
if (-not $runningOnWindows) {
    throw "This build script must run on Windows so PyInstaller can collect Windows native DLLs."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = Join-Path $repoRoot "build\pyinstaller"
$entrypoint = Join-Path $buildRoot "squisher_entrypoint.py"

Push-Location $repoRoot
try {
    if ($Clean) {
        Remove-Item -Recurse -Force "build\pyinstaller", "dist\squisher.exe" -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Force $buildRoot | Out-Null

    @'
from multiprocessing import freeze_support

from squisher import main


if __name__ == "__main__":
    freeze_support()
    main()
'@ | Set-Content -Path $entrypoint -Encoding utf8

    uv sync --locked --all-groups

    $pyinstallerArgs = @(
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name", "squisher",
        "--distpath", "dist",
        "--workpath", "build\pyinstaller\work",
        "--specpath", "build\pyinstaller\spec",
        "--collect-all", "aicspylibczi",
        "--collect-all", "imagecodecs",
        "--collect-all", "numcodecs",
        "--collect-all", "zarr",
        "--collect-all", "squisher",
        "--exclude-module", "cupy",
        "--exclude-module", "cucim",
        $entrypoint
    )

    uv run --with pyinstaller pyinstaller @pyinstallerArgs

    $exe = Join-Path $repoRoot "dist\squisher.exe"
    if (-not (Test-Path $exe)) {
        throw "Expected PyInstaller output was not created: $exe"
    }

    Write-Host "Built $exe"
}
finally {
    Pop-Location
}
