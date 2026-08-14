[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $true)]
    [string]$ExtensionV2Results,

    [string]$SourceProject = (Join-Path $PSScriptRoot ".."),

    [ValidateRange(1, 1024)]
    [int]$MaxPublicFileMB = 50
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "${Description} does not exist or is not a directory: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-DestinationOutside {
    param(
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][string]$RootPath,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $separator = [IO.Path]::DirectorySeparatorChar
    $prefix = $RootPath.TrimEnd($separator, [IO.Path]::AltDirectorySeparatorChar) + $separator
    if (
        $DestinationPath.Equals($RootPath, [StringComparison]::OrdinalIgnoreCase) -or
        $DestinationPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Destination must be outside ${Description}: $RootPath"
    }
}

function New-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required public artifact is missing: $Source"
    }
    New-Directory -Path (Split-Path -Parent $Destination)
    Copy-Item -LiteralPath $Source -Destination $Destination
}

function Copy-FilteredTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string[]]$Extensions
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        return
    }
    $reparsePoint = Get-ChildItem -LiteralPath $Source -Recurse -Force |
        Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } |
        Select-Object -First 1
    if ($null -ne $reparsePoint) {
        throw "Public source trees may not contain links or junctions: $($reparsePoint.FullName)"
    }
    $sourcePrefix = $Source.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    $destinationPrefix = [IO.Path]::GetFullPath($Destination).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    $normalizedExtensions = @($Extensions | ForEach-Object { $_.ToLowerInvariant() })
    Get-ChildItem -LiteralPath $Source -Recurse -File | ForEach-Object {
        if ($_.FullName -match '[\\/](?:__pycache__|\.pytest_cache)[\\/]') {
            return
        }
        if ($normalizedExtensions -notcontains $_.Extension.ToLowerInvariant()) {
            return
        }
        $relative = $_.FullName.Substring($sourcePrefix.Length)
        $target = [IO.Path]::GetFullPath((Join-Path $Destination $relative))
        if (-not $target.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Filtered-tree target escapes its public destination: $relative"
        }
        New-Directory -Path (Split-Path -Parent $target)
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }
}

function Assert-FileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if (-not $actual.Equals($Expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Hash mismatch for ${Description}: $Path"
    }
}

function Assert-OptionalFalseProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$PropertyName,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $property = $Object.PSObject.Properties[$PropertyName]
    if ($null -ne $property -and $property.Value -ne $false) {
        throw "${Description} must be false when present"
    }
}

function Resolve-ResultRelativePath {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)
    $normalized = $ManifestPath.Replace('/', '\')
    $prefix = 'outputs\extension-v2\'
    if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $normalized.Substring($prefix.Length)
    }
    return $normalized
}

function Resolve-ResultArtifactPath {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$ResultsRoot,
        [Parameter(Mandatory = $true)][string]$SourceRoot
    )
    $normalized = $ManifestPath.Replace('/', '\')
    if ($normalized.StartsWith('outputs\extension-v2\', [StringComparison]::OrdinalIgnoreCase)) {
        return Join-Path $ResultsRoot (Resolve-ResultRelativePath -ManifestPath $normalized)
    }
    return Join-Path $SourceRoot $normalized
}

function Assert-EvidenceManifestHashes {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$ResultsRoot,
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$PublicV2Root
    )
    $manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    foreach ($property in $manifest.input_artifacts.PSObject.Properties) {
        $entry = $property.Value
        $source = Resolve-ResultArtifactPath `
            -ManifestPath $entry.path `
            -ResultsRoot $ResultsRoot `
            -SourceRoot $SourceRoot
        Assert-FileSha256 -Path $source -Expected $entry.sha256 -Description "evidence input $($property.Name)"
    }
    foreach ($property in $manifest.outputs.PSObject.Properties) {
        $entry = $property.Value
        $source = Resolve-ResultArtifactPath `
            -ManifestPath $entry.path `
            -ResultsRoot $ResultsRoot `
            -SourceRoot $SourceRoot
        Assert-FileSha256 -Path $source -Expected $entry.sha256 -Description "evidence output $($property.Name)"
        $public = Join-Path (Join-Path $PublicV2Root "evidence-v2") (Split-Path -Leaf $entry.path)
        Assert-FileSha256 -Path $public -Expected $entry.sha256 -Description "public evidence output $($property.Name)"
    }
}

function Get-CsvColumns {
    param([Parameter(Mandatory = $true)][string]$Path)
    Add-Type -AssemblyName Microsoft.VisualBasic
    $fileStream = $null
    $gzipStream = $null
    $reader = $null
    $parser = $null
    try {
        $fileStream = [IO.File]::OpenRead($Path)
        if ($Path.EndsWith(".gz", [StringComparison]::OrdinalIgnoreCase)) {
            $gzipStream = New-Object IO.Compression.GzipStream(
                $fileStream,
                [IO.Compression.CompressionMode]::Decompress
            )
            $reader = New-Object IO.StreamReader($gzipStream, [Text.Encoding]::UTF8, $true)
        }
        else {
            $reader = New-Object IO.StreamReader($fileStream, [Text.Encoding]::UTF8, $true)
        }
        $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($reader)
        $parser.SetDelimiters(",")
        $parser.HasFieldsEnclosedInQuotes = $true
        $columns = $parser.ReadFields()
        if ($null -eq $columns -or $columns.Count -eq 0) {
            throw "Public CSV has no header: $Path"
        }
        return $columns
    }
    finally {
        if ($null -ne $parser) { $parser.Dispose() }
        if ($null -ne $reader) { $reader.Dispose() }
        if ($null -ne $gzipStream) { $gzipStream.Dispose() }
        if ($null -ne $fileStream) { $fileStream.Dispose() }
    }
}

function Assert-CsvColumns {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Expected
    )
    $actual = @(Get-CsvColumns -Path $Path)
    if (
        $actual.Count -ne $Expected.Count -or
        [string]::Join("`n", $actual) -cne [string]::Join("`n", $Expected)
    ) {
        throw "Unexpected metadata-only CSV schema: $Path"
    }
}

function Assert-NoForbiddenCsvColumns {
    param([Parameter(Mandatory = $true)][string]$Path)
    $forbidden = @(
        "func", "function", "function_text", "code", "target", "label",
        "labels", "target_label", "source_label", "conflicting_label", "vul"
    )
    $columns = @(Get-CsvColumns -Path $Path | ForEach-Object { $_.Trim().ToLowerInvariant() })
    $found = @($columns | Where-Object { $forbidden -contains $_ })
    if ($found.Count -gt 0) {
        throw "Forbidden row-level code/label column in public CSV $Path`: $($found -join ', ')"
    }
}

function Assert-NoForbiddenJsonFields {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string[]]$AllowedFields = @()
    )
    $forbidden = @(
        "func", "function", "function_text", "code", "target", "label",
        "labels", "target_label", "source_label", "conflicting_label", "vul"
    )
    $rootObject = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    $stack = New-Object Collections.Stack
    $stack.Push($rootObject)
    while ($stack.Count -gt 0) {
        $current = $stack.Pop()
        if ($null -eq $current -or $current -is [string]) {
            continue
        }
        if ($current -is [Collections.IDictionary]) {
            foreach ($key in $current.Keys) {
                $normalizedKey = $key.ToString().Trim().ToLowerInvariant()
                if ($forbidden -contains $normalizedKey -and $AllowedFields -notcontains $normalizedKey) {
                    throw "Forbidden row-level code/label field in public JSON $Path`: $key"
                }
                $stack.Push($current[$key])
            }
            continue
        }
        if ($current -is [Management.Automation.PSCustomObject]) {
            foreach ($property in $current.PSObject.Properties) {
                $normalizedName = $property.Name.Trim().ToLowerInvariant()
                if ($forbidden -contains $normalizedName -and $AllowedFields -notcontains $normalizedName) {
                    throw "Forbidden row-level code/label field in public JSON $Path`: $($property.Name)"
                }
                $stack.Push($property.Value)
            }
            continue
        }
        if ($current -is [Collections.IEnumerable]) {
            foreach ($item in $current) {
                $stack.Push($item)
            }
        }
    }
}

$templateRoot = Resolve-Directory -Path (Join-Path $PSScriptRoot "..") -Description "snapshot-template root"
$sourceRoot = Resolve-Directory -Path $SourceProject -Description "source project"
$resultsRoot = Resolve-Directory -Path $ExtensionV2Results -Description "final extension-v2 results root"
$destinationPath = [IO.Path]::GetFullPath($Destination)

foreach ($rootCheck in @(
    @($templateRoot, "snapshot-template root"),
    @($sourceRoot, "source project"),
    @($resultsRoot, "extension-v2 results root")
)) {
    Assert-DestinationOutside -DestinationPath $destinationPath -RootPath $rootCheck[0] -Description $rootCheck[1]
}
if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists: $destinationPath"
}

New-Directory -Path (Split-Path -Parent $destinationPath)
$stagingPath = "$destinationPath.building-$([Guid]::NewGuid().ToString('N'))"
New-Directory -Path $stagingPath
$buildSucceeded = $false

try {
    $configDestination = Join-Path $stagingPath "configs"
    foreach ($configName in @(
        "preregistered_experiment.json",
        "preregistered_extension_v2.json"
    )) {
        Copy-RequiredFile `
            -Source (Join-Path $sourceRoot "configs\$configName") `
            -Destination (Join-Path $configDestination $configName)
    }

    Copy-FilteredTree `
        -Source (Join-Path $sourceRoot "scripts") `
        -Destination (Join-Path $stagingPath "scripts") `
        -Extensions @(".ps1", ".py")
    Copy-FilteredTree `
        -Source (Join-Path $sourceRoot "src") `
        -Destination (Join-Path $stagingPath "src") `
        -Extensions @(".py", ".ps1")
    Copy-FilteredTree `
        -Source (Join-Path $sourceRoot "tests") `
        -Destination (Join-Path $stagingPath "tests") `
        -Extensions @(".py")

    # The hardened builder and extension-v2 README always come from this
    # staging repository, even when the scientific source tree is elsewhere.
    Copy-RequiredFile `
        -Source (Join-Path $templateRoot "src\build_public_snapshot.ps1") `
        -Destination (Join-Path $stagingPath "src\build_public_snapshot.ps1")
    foreach ($rootFile in @("README.md", ".gitignore")) {
        Copy-RequiredFile `
            -Source (Join-Path $templateRoot $rootFile) `
            -Destination (Join-Path $stagingPath $rootFile)
    }
    $requirementsSource = Join-Path $sourceRoot "requirements.txt"
    if (-not (Test-Path -LiteralPath $requirementsSource -PathType Leaf)) {
        $requirementsSource = Join-Path $templateRoot "requirements.txt"
    }
    Copy-RequiredFile `
        -Source $requirementsSource `
        -Destination (Join-Path $stagingPath "requirements.txt")

    $paperSource = $null
    foreach ($candidate in @(
        (Join-Path $sourceRoot "paper_rewriting_output\final_paper"),
        (Join-Path $sourceRoot "paper")
    )) {
        if (Test-Path -LiteralPath (Join-Path $candidate "main.tex") -PathType Leaf) {
            $paperSource = $candidate
            break
        }
    }
    if ($null -eq $paperSource) {
        throw "No final paper source containing main.tex was found under $sourceRoot"
    }
    $paperDestination = Join-Path $stagingPath "paper"
    foreach ($paperFile in @("main.tex", "references.bib", "paper.pdf")) {
        Copy-RequiredFile `
            -Source (Join-Path $paperSource $paperFile) `
            -Destination (Join-Path $paperDestination $paperFile)
    }
    foreach ($optionalPaperFile in @(
        "fcs.cls",
        "fcs.bst",
        "logo.pdf",
        "xurl.sty",
        "VulTriage_FCS_highlights.pptx"
    )) {
        $source = Join-Path $paperSource $optionalPaperFile
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-RequiredFile -Source $source -Destination (Join-Path $paperDestination $optionalPaperFile)
        }
    }
    # Figures are sourced from the finalized extension-v2 result root so the
    # public paper cannot silently retain a stale legacy figure directory.
    $paperFigureSource = Join-Path $resultsRoot "figures-v2"
    if (-not (Test-Path -LiteralPath $paperFigureSource -PathType Container)) {
        throw "Final extension-v2 figures-v2 directory is missing: $paperFigureSource"
    }
    $paperFigureDestination = Join-Path $paperDestination "figures"
    Copy-FilteredTree `
        -Source $paperFigureSource `
        -Destination $paperFigureDestination `
        -Extensions @(".pdf", ".png", ".svg", ".tiff")
    Copy-RequiredFile `
        -Source (Join-Path $paperFigureSource "figure_manifest.json") `
        -Destination (Join-Path $paperFigureDestination "figure_manifest.json")
    Copy-FilteredTree `
        -Source (Join-Path $paperFigureSource "data") `
        -Destination (Join-Path $paperFigureDestination "data") `
        -Extensions @(".csv")

    $protocolDestination = Join-Path $stagingPath "protocol"
    foreach ($protocolName in @(
        "extension_preregistration_v2.md",
        "extension_codebert_execution_amendment_v2.md",
        "extension_v2_run_plan.md",
        "near_duplicate_audit_v2.md",
        "frontiers_submission_requirements_v2.md"
    )) {
        Copy-RequiredFile `
            -Source (Join-Path $sourceRoot "research\$protocolName") `
            -Destination (Join-Path $protocolDestination $protocolName)
    }

    $publicResultsDestination = Join-Path $stagingPath "public_results"
    foreach ($v1ResultName in @(
        "analysis_manifest.json",
        "evaluation_manifest.json",
        "fold_seed_metrics.csv",
        "paired_project_comparisons.csv",
        "support_summary.csv"
    )) {
        Copy-RequiredFile `
            -Source (Join-Path $templateRoot "public_results\$v1ResultName") `
            -Destination (Join-Path $publicResultsDestination "v1\$v1ResultName")
    }

    # Every v2 artifact is selected by an exact path relative to the finalized
    # extension-v2 root. This prevents stale or aborted runs with the same
    # basename from being published accidentally.
    $v2ResultPaths = @(
        "manifest-v1\manifest_summary.json",
        "manifest-v1\extension_manifest.csv.gz",
        "near-duplicate-v1\near_duplicate_summary.json",
        "near-duplicate-v1\near_duplicate_flagged_pairs.csv.gz",
        "near-duplicate-v1\near_duplicate_exclusions.csv.gz",
        "near-duplicate-v1\near_duplicate_sensitivity_cohort.csv.gz",
        "gate-v1\gate_seal.json",
        "gate-v1\primevul_gate_development.csv",
        "evaluation-v2\evaluation_manifest.json",
        "evaluation-v2\fold_seed_metrics.csv",
        "analysis-v2\analysis_manifest.json",
        "analysis-v2\project_seed_averages.csv",
        "analysis-v2\detector_project_performance.csv",
        "analysis-v2\paired_project_comparisons.csv",
        "analysis-v2\gate_discrimination.csv",
        "calibration-size-v2\calibration_size_sensitivity.csv",
        "calibration-size-v2\calibration_size_project_summary.csv",
        "calibration-size-v2\calibration_size_aggregate_summary.csv",
        "calibration-size-v2\sensitivity_manifest.json",
        "near-duplicate-sensitivity-v2\near_duplicate_fold_seed_metrics.csv",
        "near-duplicate-sensitivity-v2\near_duplicate_project_seed_averages.csv",
        "near-duplicate-sensitivity-v2\primary_sensitivity_summary.json",
        "near-duplicate-sensitivity-v2\sensitivity_manifest.json",
        "validation-v2\artifact_validation.json",
        "validation-v2\artifact_validation.md"
    )
    foreach ($relative in $v2ResultPaths) {
        Copy-RequiredFile `
            -Source (Join-Path $resultsRoot $relative) `
            -Destination (Join-Path (Join-Path $publicResultsDestination "extension-v2") $relative)
    }
    $publicV2Root = Join-Path $publicResultsDestination "extension-v2"
    Copy-RequiredFile `
        -Source (Join-Path $resultsRoot "codebert-v1\embeddings-v1\metadata.json") `
        -Destination (Join-Path $publicV2Root "codebert-v1\embedding_metadata.json")
    Copy-RequiredFile `
        -Source (Join-Path $resultsRoot "codebert-v1\manifest_metadata.json") `
        -Destination (Join-Path $publicV2Root "codebert-v1\manifest_metadata.json")

    # Evidence and efficiency summaries are public aggregate records. Copy the
    # complete approved trees, while the later schema/denylist audit enforces
    # that no row-level code or labels enter the snapshot.
    foreach ($publicTree in @(
        "evidence-v2",
        "evidence-validation-v2",
        "efficiency-v2",
        "efficiency-validation-v2"
    )) {
        $sourceTree = Join-Path $resultsRoot $publicTree
        if (-not (Test-Path -LiteralPath $sourceTree -PathType Container)) {
            throw "Required extension-v2 public tree is missing: $sourceTree"
        }
        Copy-FilteredTree `
            -Source $sourceTree `
            -Destination (Join-Path $publicV2Root $publicTree) `
            -Extensions @(".json", ".csv", ".md")
    }

    # Validate the hash references that connect every copied v2 aggregate to
    # its sealed producer manifest. Label-bearing decision archives stay local.
    $manifestSummary = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "manifest-v1\manifest_summary.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "manifest-v1\extension_manifest.csv.gz") -Expected $manifestSummary.manifest_sha256 -Description "extension manifest"
    Assert-FileSha256 -Path (Join-Path $stagingPath "configs\preregistered_extension_v2.json") -Expected $manifestSummary.config_sha256 -Description "extension-v2 config"

    $nearDuplicateManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_summary.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_flagged_pairs.csv.gz") -Expected $nearDuplicateManifest.artifacts.flagged_pairs.sha256 -Description "near-duplicate flagged pairs"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_exclusions.csv.gz") -Expected $nearDuplicateManifest.artifacts.near_duplicate_exclusions.sha256 -Description "near-duplicate exclusions"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_sensitivity_cohort.csv.gz") -Expected $nearDuplicateManifest.artifacts.sensitivity_cohort.sha256 -Description "near-duplicate sensitivity cohort"

    $gateManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "gate-v1\gate_seal.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "gate-v1\primevul_gate_development.csv") -Expected $gateManifest.development_csv_sha256 -Description "gate development aggregates"

    $evaluationManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "evaluation-v2\evaluation_manifest.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "evaluation-v2\fold_seed_metrics.csv") -Expected $evaluationManifest.metrics_sha256 -Description "extension-v2 evaluation metrics"

    $analysisManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "analysis-v2\analysis_manifest.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "analysis-v2\project_seed_averages.csv") -Expected $analysisManifest.project_seed_averages_sha256 -Description "project seed averages"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "analysis-v2\detector_project_performance.csv") -Expected $analysisManifest.detector_project_performance_sha256 -Description "detector project performance"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "analysis-v2\paired_project_comparisons.csv") -Expected $analysisManifest.paired_project_comparisons_sha256 -Description "paired project comparisons"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "analysis-v2\gate_discrimination.csv") -Expected $analysisManifest.gate_discrimination_sha256 -Description "gate discrimination"

    $calibrationManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "calibration-size-v2\sensitivity_manifest.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "calibration-size-v2\calibration_size_sensitivity.csv") -Expected $calibrationManifest.table_sha256 -Description "calibration-size sensitivity rows"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "calibration-size-v2\calibration_size_project_summary.csv") -Expected $calibrationManifest.project_summary_sha256 -Description "calibration-size project summary"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "calibration-size-v2\calibration_size_aggregate_summary.csv") -Expected $calibrationManifest.aggregate_summary_sha256 -Description "calibration-size aggregate summary"

    $nearSensitivityManifest = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "near-duplicate-sensitivity-v2\sensitivity_manifest.json") | ConvertFrom-Json
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-sensitivity-v2\near_duplicate_fold_seed_metrics.csv") -Expected $nearSensitivityManifest.metrics_sha256 -Description "near-duplicate sensitivity rows"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-sensitivity-v2\near_duplicate_project_seed_averages.csv") -Expected $nearSensitivityManifest.project_means_sha256 -Description "near-duplicate project summary"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-sensitivity-v2\primary_sensitivity_summary.json") -Expected $nearSensitivityManifest.primary_summary_sha256 -Description "near-duplicate primary summary"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "evaluation-v2\evaluation_manifest.json") -Expected $nearSensitivityManifest.evaluation_manifest_sha256 -Description "sensitivity evaluation manifest"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_summary.json") -Expected $nearSensitivityManifest.near_duplicate_summary_sha256 -Description "sensitivity near-duplicate manifest"
    Assert-FileSha256 -Path (Join-Path $publicV2Root "near-duplicate-v1\near_duplicate_sensitivity_cohort.csv.gz") -Expected $nearSensitivityManifest.cohort_sha256 -Description "sensitivity retained cohort"

    $artifactValidation = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "validation-v2\artifact_validation.json") | ConvertFrom-Json
    Assert-OptionalFalseProperty `
        -Object $artifactValidation `
        -PropertyName "target_vulnerability_labels_accessed" `
        -Description "artifact validation target-label access flag"
    if (
        $artifactValidation.status -ne "PASS" -or
        [int]$artifactValidation.projects -ne 24 -or
        [int]$artifactValidation.project_rows -ne 10800 -or
        [int]$artifactValidation.main_metric_rows -ne 54000 -or
        [int]$artifactValidation.calibration_sensitivity_rows -ne 14400 -or
        [int]$artifactValidation.near_duplicate_sensitivity_rows -ne 6480 -or
        [int]$artifactValidation.methods -ne 25 -or
        [int]$artifactValidation.risk_grid_cells -ne 9 -or
        @($artifactValidation.seeds).Count -ne 5 -or
        [int]$artifactValidation.figure_assets -ne 20 -or
        [int]$artifactValidation.figure_data_assets -ne 5
    ) {
        throw "Extension-v2 independent artifact validation did not pass"
    }
    foreach ($sealReference in @(
        @("hashing", "predictions\hashing-v8\prediction_seal.json", $artifactValidation.prediction_seals.hashing.sha256, "independent"),
        @("CodeBERT", "predictions\codebert-v2\prediction_seal.json", $artifactValidation.prediction_seals.codebert.sha256, "deterministic_liblinear_replicates")
    )) {
        $sealPath = Join-Path $resultsRoot $sealReference[1]
        Assert-FileSha256 -Path $sealPath -Expected $sealReference[2] -Description "$($sealReference[0]) prediction seal"
        $seal = Get-Content -Raw -LiteralPath $sealPath | ConvertFrom-Json
        if (
            $seal.config_sha256 -ne $artifactValidation.config_sha256 -or
            $seal.target_label_vault_argument_present -ne $false -or
            $seal.target_vulnerability_labels_accessed -ne $false -or
            $seal.seed_reuse_mode -ne $sealReference[3] -or
            @($seal.prediction_files.PSObject.Properties).Count -ne 240
        ) {
            throw "Invalid sealed prediction metadata for $($sealReference[0])"
        }
    }

    $embeddingMetadata = Get-Content -Raw -LiteralPath (Join-Path $publicV2Root "codebert-v1\embedding_metadata.json") | ConvertFrom-Json
    $expectedCodeBertRevision = "3b0952feddeffad0063f274080e3c23d75e7eb39"
    if (
        $embeddingMetadata.labels_used -ne $false -or
        $embeddingMetadata.model -ne "microsoft/codebert-base" -or
        $embeddingMetadata.revision -cne $expectedCodeBertRevision
    ) {
        throw "CodeBERT metadata is not label-free or differs from the frozen model revision"
    }
    Assert-FileSha256 -Path (Join-Path $publicV2Root "codebert-v1\manifest_metadata.json") -Expected $embeddingMetadata.manifest_metadata_sha256 -Description "CodeBERT input manifest metadata"

    $figureManifest = Get-Content -Raw -LiteralPath (Join-Path $paperFigureDestination "figure_manifest.json") | ConvertFrom-Json
    if (
        @($figureManifest.figures).Count -ne 5 -or
        @($figureManifest.assets.PSObject.Properties).Count -ne 20 -or
        @($figureManifest.data_assets.PSObject.Properties).Count -ne 5
    ) {
        throw "Final extension-v2 figure manifest has unexpected asset counts"
    }
    foreach ($property in $figureManifest.assets.PSObject.Properties) {
        Assert-FileSha256 `
            -Path (Join-Path $paperFigureSource $property.Name) `
            -Expected $property.Value.sha256 `
            -Description "source figure $($property.Name)"
        Assert-FileSha256 `
            -Path (Join-Path $paperFigureDestination $property.Name) `
            -Expected $property.Value.sha256 `
            -Description "public figure $($property.Name)"
    }
    foreach ($property in $figureManifest.data_assets.PSObject.Properties) {
        Assert-FileSha256 `
            -Path (Join-Path (Join-Path $paperFigureSource "data") $property.Name) `
            -Expected $property.Value.sha256 `
            -Description "source figure data $($property.Name)"
        Assert-FileSha256 `
            -Path (Join-Path (Join-Path $paperFigureDestination "data") $property.Name) `
            -Expected $property.Value.sha256 `
            -Description "public figure data $($property.Name)"
    }

    $evidenceManifestPath = Join-Path $publicV2Root "evidence-v2\evidence_manifest.json"
    Assert-EvidenceManifestHashes `
        -ManifestPath $evidenceManifestPath `
        -ResultsRoot $resultsRoot `
        -SourceRoot $sourceRoot `
        -PublicV2Root $publicV2Root
    $evidenceManifest = Get-Content -Raw -LiteralPath $evidenceManifestPath | ConvertFrom-Json
    if (
        $evidenceManifest.status -ne "complete" -or
        $evidenceManifest.target_vulnerability_labels_accessed -ne $false -or
        [int]$evidenceManifest.counts.projects -ne 24 -or
        [int]$evidenceManifest.counts.main_metric_rows -ne 54000
    ) {
        throw "Extension-v2 evidence manifest is incomplete or label-bearing"
    }
    $evidenceValidationPath = Join-Path $publicV2Root "evidence-validation-v2\evidence_validation.json"
    $evidenceValidation = Get-Content -Raw -LiteralPath $evidenceValidationPath | ConvertFrom-Json
    if (
        $evidenceValidation.status -ne "PASS" -or
        $evidenceValidation.target_vulnerability_labels_accessed -ne $false -or
        [int]$evidenceValidation.projects -ne 24 -or
        [int]$evidenceValidation.main_metric_rows -ne 54000
    ) {
        throw "Extension-v2 evidence validation did not pass"
    }
    Assert-FileSha256 `
        -Path $evidenceManifestPath `
        -Expected $evidenceValidation.evidence_manifest_sha256 `
        -Description "public evidence manifest"
    Assert-FileSha256 `
        -Path (Join-Path $resultsRoot "evidence-v2\evidence_manifest.json") `
        -Expected $evidenceValidation.evidence_manifest_sha256 `
        -Description "source evidence manifest"

    $efficiencyManifestPath = Join-Path $publicV2Root "efficiency-v2\efficiency_manifest.json"
    $efficiencyManifest = Get-Content -Raw -LiteralPath $efficiencyManifestPath | ConvertFrom-Json
    if (
        $efficiencyManifest.protocol_version -ne "vultriage-extension-v2" -or
        $efficiencyManifest.target_vulnerability_labels_accessed -ne $false -or
        [int]$efficiencyManifest.summary_rows -ne 2 -or
        [int]$efficiencyManifest.executed_head_fit_rows -ne 144
    ) {
        throw "Extension-v2 efficiency manifest is incomplete or label-bearing"
    }
    foreach ($reference in @(
        @("CodeBERT embedding metadata", "codebert-v1\embeddings-v1\metadata.json", $efficiencyManifest.codebert_embedding_metadata_sha256),
        @("CodeBERT part runtime", "efficiency-v2\codebert_part_runtime.csv", $efficiencyManifest.codebert_part_runtime_sha256),
        @("detector efficiency summary", "efficiency-v2\detector_efficiency_summary.csv", $efficiencyManifest.detector_efficiency_summary_sha256),
        @("executed head fits", "efficiency-v2\executed_head_fits.csv", $efficiencyManifest.executed_head_fits_sha256),
        @("source package summary", "source-v2\package_summary.json", $efficiencyManifest.package_summary_sha256),
        @("efficiency script", "..\src\analyze_extension_v2_efficiency.py", $efficiencyManifest.script_sha256)
    )) {
        $referencePath = if ($reference[1].StartsWith("..\")) {
            Join-Path $sourceRoot ($reference[1].Substring(3))
        }
        else {
            Join-Path $resultsRoot $reference[1]
        }
        Assert-FileSha256 -Path $referencePath -Expected $reference[2] -Description $reference[0]
    }
    foreach ($property in $efficiencyManifest.codebert_part_seals.PSObject.Properties) {
        $partSealPath = Join-Path $resultsRoot (Resolve-ResultRelativePath -ManifestPath $property.Name)
        Assert-FileSha256 -Path $partSealPath -Expected $property.Value -Description "CodeBERT part seal $($property.Name)"
    }
    if ($efficiencyManifest.prediction_seals.hashing -ne $artifactValidation.prediction_seals.hashing.sha256 -or
        $efficiencyManifest.prediction_seals.codebert -ne $artifactValidation.prediction_seals.codebert.sha256) {
        throw "Efficiency and artifact validation prediction seals disagree"
    }
    $efficiencyValidationPath = Join-Path $publicV2Root "efficiency-validation-v2\efficiency_validation.json"
    $efficiencyValidation = Get-Content -Raw -LiteralPath $efficiencyValidationPath | ConvertFrom-Json
    if (
        $efficiencyValidation.status -ne "PASS" -or
        $efficiencyValidation.target_vulnerability_labels_accessed -ne $false -or
        [int]$efficiencyValidation.projects -ne 24 -or
        [int]$efficiencyValidation.summary_rows -ne 2 -or
        [int]$efficiencyValidation.executed_head_fit_rows -ne 144
    ) {
        throw "Extension-v2 efficiency validation did not pass"
    }
    Assert-FileSha256 `
        -Path $efficiencyManifestPath `
        -Expected $efficiencyValidation.efficiency_manifest_sha256 `
        -Description "public efficiency manifest"

    $metadataSchemas = @(
        @{
            Path = "manifest-v1\extension_manifest.csv.gz"
            Columns = @("position", "row_id", "dataset", "source_file", "line_number", "project", "project_group", "commit_id", "exact_code_key")
        },
        @{
            Path = "near-duplicate-v1\near_duplicate_flagged_pairs.csv.gz"
            Columns = @("target_row_id", "target_project", "target_project_group", "target_source_file", "target_line_number", "target_exact_code_key", "target_token_count", "prime_row_id", "prime_project", "prime_project_group", "prime_source_file", "prime_line_number", "prime_exact_code_key", "prime_token_count", "intersection_count", "union_count", "exact_jaccard", "minhash_agreement", "minhash_estimate")
        },
        @{
            Path = "near-duplicate-v1\near_duplicate_exclusions.csv.gz"
            Columns = @("position", "target_row_id", "target_project", "target_project_group", "target_source_file", "target_line_number", "target_exact_code_key", "target_token_count", "flagged_prime_pair_count", "maximum_exact_jaccard")
        },
        @{
            Path = "near-duplicate-v1\near_duplicate_sensitivity_cohort.csv.gz"
            Columns = @("position", "row_id", "dataset", "source_file", "line_number", "project", "project_group", "commit_id", "exact_code_key")
        }
    )
    foreach ($schema in $metadataSchemas) {
        Assert-CsvColumns -Path (Join-Path $publicV2Root $schema.Path) -Expected $schema.Columns
    }

    foreach ($csvRoot in @(
        $publicResultsDestination,
        (Join-Path $paperFigureDestination "data")
    )) {
        if (Test-Path -LiteralPath $csvRoot -PathType Container) {
            Get-ChildItem -LiteralPath $csvRoot -Recurse -File |
                Where-Object { $_.Name -match '(?i)\.csv(?:\.gz)?$' } |
                ForEach-Object { Assert-NoForbiddenCsvColumns -Path $_.FullName }
        }
    }
    foreach ($jsonRoot in @($publicResultsDestination, $paperFigureDestination)) {
        Get-ChildItem -LiteralPath $jsonRoot -Recurse -Filter "*.json" -File |
            ForEach-Object { Assert-NoForbiddenJsonFields -Path $_.FullName }
    }
    Assert-NoForbiddenJsonFields -Path (Join-Path $stagingPath "configs\preregistered_experiment.json")
    Assert-NoForbiddenJsonFields `
        -Path (Join-Path $stagingPath "configs\preregistered_extension_v2.json") `
        -AllowedFields @("label")

    $validationDestination = Join-Path $stagingPath "validation"
    $validationNames = @(
        "results_validation.md",
        "integrity_audit.md",
        "citation_support_bank.md",
        "claim_register.md",
        "evidence_bank.md",
        "confirmed_contribution.md",
        "reviewer_audit.md"
    ) | Select-Object -Unique
    foreach ($validationName in $validationNames) {
        $source = Join-Path (Join-Path $sourceRoot "paper_rewriting_output") $validationName
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            $source = Join-Path (Join-Path $templateRoot "validation") $validationName
        }
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-RequiredFile -Source $source -Destination (Join-Path $validationDestination $validationName)
        }
    }
    $citationReport = Get-ChildItem `
        -LiteralPath (Join-Path $sourceRoot "paper_rewriting_output") `
        -Recurse `
        -Filter "citation_verification_final.html" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -ne $citationReport) {
        Copy-RequiredFile `
            -Source $citationReport.FullName `
            -Destination (Join-Path $validationDestination "citation_verification_final.html")
    }

    $maximumBytes = [int64]$MaxPublicFileMB * 1MB
    $forbiddenExtensions = @(
        ".sqlite", ".sqlite3", ".db", ".db3", ".npz", ".npy",
        ".pt", ".bin", ".safetensors", ".joblib", ".pkl", ".pickle", ".jsonl"
    )
    $forbiddenName = '(?i)(?:label[_-]?vault|extension[_-]?labels|source[_-]?labels|target[_-]?labels|diversevul_20230702|primevul_(?:train|valid|test))'
    $forbiddenDirectory = '(?i)(^|[\\/])(?:data[\\/]external|data[\\/]cache|outputs|checkpoints|hf-cache|predictions?|decisions?|embeddings?)([\\/]|$)'
    $stagingPrefix = $stagingPath.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    Get-ChildItem -LiteralPath $stagingPath -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($stagingPrefix.Length)
        if ($_.Length -gt $maximumBytes) {
            throw "Unexpected public file larger than $MaxPublicFileMB MB: $relative"
        }
        if ($forbiddenExtensions -contains $_.Extension.ToLowerInvariant()) {
            throw "Forbidden binary/private artifact in public snapshot: $relative"
        }
        if ($_.Name -match $forbiddenName) {
            throw "Forbidden dataset/label artifact in public snapshot: $relative"
        }
        if ($relative -match $forbiddenDirectory) {
            throw "Forbidden private/cache directory in public snapshot: $relative"
        }
    }

    $manifestEntries = @(
        Get-ChildItem -LiteralPath $stagingPath -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = $_.FullName.Substring($stagingPrefix.Length).Replace('\', '/')
                bytes = $_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
            }
        }
    )
    $snapshotManifest = [ordered]@{
        schema_version = "vultriage-public-snapshot-extension-v2-v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        source_project_name = Split-Path -Leaf $sourceRoot
        extension_v2_results_name = Split-Path -Leaf $resultsRoot
        file_inventory_scope = "all staged public files except this manifest"
        excluded_classes = @(
            "raw PrimeVul and DiverseVul releases",
            "all label vaults and label packages",
            "CodeBERT embedding and feature caches",
            "SQLite and other private working indexes",
            "per-function prediction and decision archives"
        )
        files = $manifestEntries
    }
    $snapshotManifestPath = Join-Path $stagingPath "public_snapshot_manifest.json"
    $snapshotManifestJson = $snapshotManifest | ConvertTo-Json -Depth 6
    $utf8NoBom = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($snapshotManifestPath, $snapshotManifestJson + "`n", $utf8NoBom)
    if ((Get-Item -LiteralPath $snapshotManifestPath).Length -gt $maximumBytes) {
        throw "Generated public snapshot manifest exceeds $MaxPublicFileMB MB"
    }

    Move-Item -LiteralPath $stagingPath -Destination $destinationPath
    $buildSucceeded = $true
    Write-Output $destinationPath
}
finally {
    if (-not $buildSucceeded -and (Test-Path -LiteralPath $stagingPath)) {
        Remove-Item -LiteralPath $stagingPath -Recurse -Force
    }
}
