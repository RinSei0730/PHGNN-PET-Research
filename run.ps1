param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [Parameter(Mandatory = $true)][string]$CohortPath,
    [string]$OutputPath = 'results/new_round',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$studyPath = $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Python executable does not exist.' }
if (-not (Test-Path -LiteralPath $CohortPath -PathType Leaf)) { throw 'Prepared cohort file does not exist.' }
$pythonResolved = (Resolve-Path -LiteralPath $PythonPath).Path
$cohortResolved = (Resolve-Path -LiteralPath $CohortPath).Path
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) { $OutputPath = Join-Path $studyPath $OutputPath }
$runArguments = @('-X', 'utf8', '-B', '-u', (Join-Path $studyPath 'src/run_experiments.py'), '--cohort', $cohortResolved, '--out', $OutputPath)
if ($Resume) { $runArguments += '--resume' }
& $pythonResolved @runArguments
$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0) { throw ('Experiment exited with code ' + $runExitCode) }
