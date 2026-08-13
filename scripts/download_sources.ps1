$ErrorActionPreference = 'Stop'

$sourceDir = Join-Path $PSScriptRoot '..\research\sources'
New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null

$items = @(
    @('feng2020_codebert.pdf', 'https://aclanthology.org/2020.findings-emnlp.139.pdf'),
    @('fu2022_linevul.pdf', 'https://arxiv.org/pdf/2204.05106'),
    @('ding2025_primevul.pdf', 'https://arxiv.org/pdf/2403.18624'),
    @('chen2023_diversevul.pdf', 'https://arxiv.org/pdf/2304.00409'),
    @('nguyen2024_dam2p.pdf', 'https://research.monash.edu/files/623215953/612341622-oa.pdf'),
    @('haque2026_zsvuld.pdf', 'https://pure.ulster.ac.uk/ws/portalfiles/portal/235264532/s10664-025-10749-4.cleaned_1_.pdf'),
    @('wang2025_prom.pdf', 'https://zwang4.github.io/publications/cgo25.pdf'),
    @('li2021_program_shift.pdf', 'https://arxiv.org/pdf/2107.10989'),
    @('rathnasuriya2026_defer.pdf', 'https://arxiv.org/pdf/2605.19369'),
    @('guo2017_calibration.pdf', 'https://proceedings.mlr.press/v70/guo17a/guo17a.pdf'),
    @('geifman2019_selectivenet.pdf', 'https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf'),
    @('tibshirani2019_covariate_shift.pdf', 'https://proceedings.neurips.cc/paper_files/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf'),
    @('angelopoulos2024_crc.pdf', 'https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf'),
    @('farinhas2024_necrc.pdf', 'https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf'),
    @('almeida2025_hpcrc.pdf', 'https://raw.githubusercontent.com/mlresearch/v266/main/assets/almeida25a/almeida25a.pdf')
)

foreach ($item in $items) {
    $outputPath = Join-Path $sourceDir $item[0]
    try {
        Invoke-WebRequest -Uri $item[1] -OutFile $outputPath -MaximumRedirection 8 -Headers @{'User-Agent' = 'Mozilla/5.0'}
        $file = Get-Item -LiteralPath $outputPath
        $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
        $status = if ($file.Length -ge 10000) { 'OK' } else { 'WARN_TOO_SMALL' }
        "$status`t$($item[0])`t$($file.Length)`t$hash"
    }
    catch {
        "ERR`t$($item[0])`t$($_.Exception.Message)"
    }
}
