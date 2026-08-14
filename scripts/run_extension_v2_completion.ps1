param(
    [string]$WaitForPids = "",
    [string]$HashingOutput = "outputs/extension-v2/predictions/hashing-v8",
    [string]$CodeBertOutput = "outputs/extension-v2/predictions/codebert-v2",
    [string]$EvaluationOutput = "outputs/extension-v2/evaluation-v2",
    [string]$AnalysisOutput = "outputs/extension-v2/analysis-v2",
    [string]$CalibrationOutput = "outputs/extension-v2/calibration-size-v2",
    [string]$NearDuplicateOutput = "outputs/extension-v2/near-duplicate-sensitivity-v2",
    [string]$FigureOutput = "outputs/extension-v2/figures-v2"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

if ($WaitForPids) {
    $processIds = @(
        $WaitForPids.Split(",") |
            ForEach-Object { [int]$_.Trim() } |
            Where-Object { $_ -gt 0 }
    )
    Write-Output "[$(Get-Date -Format o)] Waiting for prediction processes: $($processIds -join ', ')"
    Wait-Process -Id $processIds
    Start-Sleep -Seconds 10
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    Write-Output "[$(Get-Date -Format o)] START $Name"
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format o)] COMPLETE $Name"
}

$parts = @(
    "outputs/extension-v2/predictions/codebert-v2-balanced-parts/part-a",
    "outputs/extension-v2/predictions/codebert-v2-balanced-parts/part-b",
    "outputs/extension-v2/predictions/codebert-v2-balanced-parts/part-c",
    "outputs/extension-v2/predictions/codebert-v2-balanced-parts/part-d"
)
foreach ($part in $parts) {
    if (-not (Test-Path -LiteralPath (Join-Path $part "prediction_seal.json"))) {
        throw "CodeBERT part is not sealed: $part"
    }
}

Invoke-PythonStep -Name "merge CodeBERT prediction parts" -Arguments @(
    "src/merge_prediction_parts.py",
    "--parts", $parts[0], $parts[1], $parts[2], $parts[3],
    "--output", $CodeBertOutput,
    "--detector", "codebert",
    "--config", "configs/preregistered_extension_v2.json",
    "--source-metadata", "outputs/extension-v2/source-v2/source_metadata.csv.gz",
    "--target-metadata", "outputs/extension-v2/source-v2/target_metadata.csv.gz",
    "--package-summary", "outputs/extension-v2/source-v2/package_summary.json"
)

Invoke-PythonStep -Name "run sealed hashing predictions" -Arguments @(
    "src/run_extension_predict.py",
    "--detector", "hashing",
    "--inputs", "outputs/extension-v2/source-v2",
    "--source-metadata", "outputs/extension-v2/source-v2/source_metadata.csv.gz",
    "--target-metadata", "outputs/extension-v2/source-v2/target_metadata.csv.gz",
    "--config", "configs/preregistered_extension_v2.json",
    "--output", $HashingOutput,
    "--source-features", "data/cache/hashing-v1",
    "--target-features", "outputs/extension-v2/hashing-target-v1"
)

Invoke-PythonStep -Name "evaluate sealed external predictions" -Arguments @(
    "src/evaluate_extension_v2.py",
    "--hashing-predictions", $HashingOutput,
    "--codebert-predictions", $CodeBertOutput,
    "--inputs", "outputs/extension-v2/source-v2",
    "--source-metadata", "outputs/extension-v2/source-v2/source_metadata.csv.gz",
    "--target-metadata", "outputs/extension-v2/source-v2/target_metadata.csv.gz",
    "--label-vault", "outputs/extension-v2/manifest-v1/extension_labels.csv.gz",
    "--config", "configs/preregistered_extension_v2.json",
    "--gate", "outputs/extension-v2/gate-v1",
    "--output", $EvaluationOutput
)

Invoke-PythonStep -Name "analyze project-level external results" -Arguments @(
    "src/analyze_extension_v2.py",
    "--metrics", (Join-Path $EvaluationOutput "fold_seed_metrics.csv"),
    "--evaluation-manifest", (Join-Path $EvaluationOutput "evaluation_manifest.json"),
    "--config", "configs/preregistered_extension_v2.json",
    "--output", $AnalysisOutput
)

Invoke-PythonStep -Name "analyze calibration-size sensitivity" -Arguments @(
    "src/analyze_calibration_sensitivity.py",
    "--hashing-predictions", $HashingOutput,
    "--codebert-predictions", $CodeBertOutput,
    "--inputs", "outputs/extension-v2/source-v2",
    "--source-metadata", "outputs/extension-v2/source-v2/source_metadata.csv.gz",
    "--target-metadata", "outputs/extension-v2/source-v2/target_metadata.csv.gz",
    "--label-vault", "outputs/extension-v2/manifest-v1/extension_labels.csv.gz",
    "--config", "configs/preregistered_extension_v2.json",
    "--gate", "outputs/extension-v2/gate-v1",
    "--output", $CalibrationOutput
)

Invoke-PythonStep -Name "analyze near-duplicate sensitivity" -Arguments @(
    "src/analyze_near_duplicate_sensitivity_v2.py",
    "--evaluation", $EvaluationOutput,
    "--cohort", "outputs/extension-v2/near-duplicate-v1/near_duplicate_sensitivity_cohort.csv.gz",
    "--near-duplicate-summary", "outputs/extension-v2/near-duplicate-v1/near_duplicate_summary.json",
    "--config", "configs/preregistered_extension_v2.json",
    "--output", $NearDuplicateOutput
)

Invoke-PythonStep -Name "generate extension-v2 figures" -Arguments @(
    "src/make_extension_v2_figures.py",
    "--metrics", (Join-Path $EvaluationOutput "fold_seed_metrics.csv"),
    "--project-means", (Join-Path $AnalysisOutput "project_seed_averages.csv"),
    "--gate-discrimination", (Join-Path $AnalysisOutput "gate_discrimination.csv"),
    "--gate-seal", "outputs/extension-v2/gate-v1/gate_seal.json",
    "--calibration-project-summary", "outputs/extension-v2/calibration-size-v2/calibration_size_project_summary.csv",
    "--output", $FigureOutput
)

Invoke-PythonStep -Name "validate finalized extension-v2 artifacts" -Arguments @(
    "src/validate_extension_v2_artifacts.py",
    "--root", "outputs/extension-v2",
    "--config", "configs/preregistered_extension_v2.json",
    "--output", "outputs/extension-v2/validation-v2"
)

Invoke-PythonStep -Name "build extension-v2 paper evidence" -Arguments @(
    "src/build_extension_v2_evidence.py",
    "--root", "outputs/extension-v2",
    "--config", "configs/preregistered_extension_v2.json",
    "--output", "outputs/extension-v2/evidence-v2"
)

Invoke-PythonStep -Name "validate extension-v2 paper evidence" -Arguments @(
    "src/validate_extension_v2_evidence.py",
    "--root", "outputs/extension-v2",
    "--config", "configs/preregistered_extension_v2.json",
    "--evidence", "outputs/extension-v2/evidence-v2",
    "--output", "outputs/extension-v2/evidence-validation-v2"
)

Write-Output "[$(Get-Date -Format o)] EXTENSION_V2_COMPLETION_SUCCEEDED"
