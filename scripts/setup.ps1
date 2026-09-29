param([ValidateSet('dev', 'detection', 'full')][string]$Mode = 'dev')
$ErrorActionPreference = 'Stop'
$pipelineRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:UV_CACHE_DIR = Join-Path $pipelineRoot '.uv-cache'
$env:UV_LINK_MODE = 'copy'
$pipelinePython = Join-Path $pipelineRoot 'runtime/.venv/Scripts/python.exe'
$pipelineFix = Join-Path $PSScriptRoot 'fix_windows_pth.py'
if (Test-Path -LiteralPath $pipelinePython) {
    & $pipelinePython -X utf8 $pipelineFix (Join-Path $pipelineRoot 'runtime/.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$uvArgs = @('sync', '--project', (Join-Path $pipelineRoot 'runtime'), '--locked', '--extra', 'dev', '--python', '3.10')
if ($Mode -eq 'detection') { $uvArgs += @('--extra', 'detection') }
if ($Mode -eq 'full') { $uvArgs += @('--extra', 'models') }
& uv @uvArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pipelinePython -X utf8 $pipelineFix (Join-Path $pipelineRoot 'runtime/.venv')
exit $LASTEXITCODE
