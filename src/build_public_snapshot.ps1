param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destinationPath = [IO.Path]::GetFullPath($Destination)
if ($destinationPath -eq $root -or $destinationPath.StartsWith($root + [IO.Path]::DirectorySeparatorChar)) {
    throw "Destination must be outside the source project."
}
if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists: $destinationPath"
}
New-Item -ItemType Directory -Path $destinationPath | Out-Null

foreach ($directory in @("configs", "scripts", "src", "tests")) {
    Copy-Item -Recurse -LiteralPath (Join-Path $root $directory) -Destination (Join-Path $destinationPath $directory)
}
Copy-Item -LiteralPath (Join-Path $root "README.md") -Destination $destinationPath
Copy-Item -LiteralPath (Join-Path $root ".gitignore") -Destination $destinationPath
Copy-Item -LiteralPath (Join-Path $root "requirements.txt") -Destination $destinationPath

$paperDestination = Join-Path $destinationPath "paper"
New-Item -ItemType Directory -Path $paperDestination | Out-Null
foreach ($file in @("main.tex", "references.bib", "paper.pdf")) {
    Copy-Item -LiteralPath (Join-Path $root "paper_rewriting_output\final_paper\$file") -Destination $paperDestination
}
$figureSource = Join-Path $root "paper_rewriting_output\final_paper\figures"
$figureDestination = Join-Path $paperDestination "figures"
New-Item -ItemType Directory -Path $figureDestination | Out-Null
foreach ($pattern in @("*.pdf", "*.png", "*.svg", "figure_manifest.json")) {
    Get-ChildItem -LiteralPath $figureSource -Filter $pattern -File |
        Copy-Item -Destination $figureDestination
}
Copy-Item -Recurse -LiteralPath (Join-Path $figureSource "data") -Destination (Join-Path $figureDestination "data")

$resultsDestination = Join-Path $destinationPath "public_results"
New-Item -ItemType Directory -Path $resultsDestination | Out-Null
Copy-Item -LiteralPath (Join-Path $root "outputs\exp-e1-cpu-full-analysis-v2\analysis_manifest.json") -Destination $resultsDestination
Copy-Item -LiteralPath (Join-Path $root "outputs\exp-e1-cpu-full-analysis-v2\paired_project_comparisons.csv") -Destination $resultsDestination
Copy-Item -LiteralPath (Join-Path $root "outputs\exp-e1-cpu-full-analysis-v2\support_summary.csv") -Destination $resultsDestination
Copy-Item -LiteralPath (Join-Path $root "outputs\exp-e1-cpu-full-evaluation\evaluation_manifest.json") -Destination $resultsDestination
Copy-Item -LiteralPath (Join-Path $root "outputs\exp-e1-cpu-full-evaluation\fold_seed_metrics.csv") -Destination $resultsDestination

$validationDestination = Join-Path $destinationPath "validation"
New-Item -ItemType Directory -Path $validationDestination | Out-Null
foreach ($file in @("results_validation.md", "integrity_audit.md", "citation_support_bank.md", "claim_register.md", "evidence_bank.md")) {
    Copy-Item -LiteralPath (Join-Path $root "paper_rewriting_output\$file") -Destination $validationDestination
}
Copy-Item -LiteralPath (Join-Path $root "paper_rewriting_output\reports\2026-08-13\citation_verification_final.html") -Destination $validationDestination

Get-ChildItem -LiteralPath $destinationPath -Recurse -File |
    Where-Object { $_.Length -gt 50MB } |
    ForEach-Object { throw "Unexpected file larger than 50 MB: $($_.FullName)" }

Write-Output $destinationPath
