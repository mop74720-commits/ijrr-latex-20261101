[CmdletBinding()]
param(
    [string]$XeLaTeXCommand = $env:IJRR_XELATEX,
    [string]$BibTeXCommand = $env:IJRR_BIBTEX
)

$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$ErrorActionPreference = "Stop"

function Resolve-BuildTool {
    param([string]$Name, [string]$Explicit, [string[]]$Candidates)
    if ($Explicit) { $Candidates = @($Explicit) }
    foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
        $tool = Get-Command -Name $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($tool) { return $tool.Source }
    }
    throw "$Name is missing. Install it on PATH or provide its explicit build parameter."
}

$latexCandidates = @("xelatex")
if ($env:LOCALAPPDATA) {
    $latexCandidates += Join-Path $env:LOCALAPPDATA "Programs/MiKTeX/miktex/bin/x64/xelatex.exe"
}
$latexEngine = Resolve-BuildTool -Name "XeLaTeX" -Explicit $XeLaTeXCommand -Candidates $latexCandidates
$bibCandidates = @("bibtex", (Join-Path (Split-Path $latexEngine) "bibtex.exe"))
$bibliographyEngine = Resolve-BuildTool -Name "BibTeX" -Explicit $BibTeXCommand -Candidates $bibCandidates

Push-Location $PSScriptRoot
try {
    & $latexEngine -interaction=nonstopmode -halt-on-error "main.tex"
    if ($LASTEXITCODE -ne 0) { throw "Initial XeLaTeX pass failed." }
    & $bibliographyEngine "main"
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed." }
    1..3 | ForEach-Object {
        & $latexEngine -interaction=nonstopmode -halt-on-error "main.tex"
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass failed." }
    }
    Write-Host "Built main.pdf"
} finally {
    Pop-Location
}
