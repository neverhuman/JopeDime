#Requires -Version 5.1
<#
.SYNOPSIS
  Carbon Zero-Knowledge Serverless CLI Installer for Windows (x86_64 & ARM64).
.DESCRIPTION
  Verifies the embedded owner NIST P-256 SPKI fingerprint, verifies the detached ECDSA-P256
  signature over SHA256SUMS.txt using .NET Cryptography (zero external OpenSSL dependency),
  validates SHA-256 digests of carbon-windows-<arch>-msvc.zip and carbon-clouds.json, and
  installs carbon.exe into $env:LOCALAPPDATA\Programs\carbon\bin.
.EXAMPLE
  irm https://downloads.neverhuman.org/carbon/install.ps1 | iex
#>
[CmdletBinding()]
param(
  [string]$Version = "carbon-cli-v1.1.0",
  [string]$BinDir = "",
  [string]$ConfigDir = "",
  [string]$BaseUrl = "",
  [string]$SourceDir = "",
  [string]$AllowTestPubkey = "",
  [switch]$SkipPing
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SigningPubFingerprint = "de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e"
$EmbeddedSigningPubPem = @"
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMx44DTDE4ObFoh7otiKzAJA9fd8U
eEunkkDgvRCL74VyxIW9Q0wt49FMtTqv7aH7sj//B2uqMwPdm/MlBu45Nw==
-----END PUBLIC KEY-----
"@ + "`n"

function Write-CarbonLog([string]$Message) {
  [Console]::Error.WriteLine("[carbon-install] $Message")
}

if ([string]::IsNullOrWhiteSpace($BinDir)) {
  $LocalApp = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
  $BinDir = Join-Path $LocalApp "Programs\carbon\bin"
}
if ([string]::IsNullOrWhiteSpace($ConfigDir)) {
  $HomeRoot = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
  $ConfigDir = Join-Path $HomeRoot ".config\jopedime"
}

$ArchRaw = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
switch ($ArchRaw) {
  "x64"   { $BaseAsset = "carbon-windows-x86_64-msvc" }
  "arm64" { $BaseAsset = "carbon-windows-aarch64-msvc" }
  default { throw "[carbon-install] ERROR: Unsupported Windows CPU architecture '$ArchRaw' (expected X64 or Arm64)" }
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  $EffectiveBaseUrl = "https://github.com/neverhuman/JopeDime/releases/download/$Version"
  $UseSecondaryMirror = $true
} else {
  $EffectiveBaseUrl = $BaseUrl.TrimEnd("/")
  $UseSecondaryMirror = $false
}
$MirrorUrl = "https://downloads.neverhuman.org/carbon/releases/$Version"

$StageDir = Join-Path ([System.IO.Path]::GetTempPath()) ("carbon-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

function Get-FileSha256Lower([string]$Path) {
  return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Fetch-CarbonAsset([string]$AssetName, [string]$DestPath) {
  if (-not [string]::IsNullOrWhiteSpace($SourceDir)) {
    $Src = Join-Path $SourceDir $AssetName
    if (-not (Test-Path -LiteralPath $Src)) {
      throw "[carbon-install] ERROR: Missing $Src"
    }
    Copy-Item -LiteralPath $Src -Destination $DestPath -Force
    return
  }
  $Primary = "$EffectiveBaseUrl/$AssetName"
  try {
    Invoke-WebRequest -Uri $Primary -OutFile $DestPath -UseBasicParsing -TimeoutSec 30
  } catch {
    if (-not $UseSecondaryMirror) { throw }
    Write-CarbonLog "Primary download unavailable for $AssetName; trying official mirror ($MirrorUrl)..."
    Invoke-WebRequest -Uri "$MirrorUrl/$AssetName" -OutFile $DestPath -UseBasicParsing -TimeoutSec 30
  }
}

function Convert-Asn1DerEcdsaToP1363([byte[]]$DerSig) {
  if ($DerSig.Length -lt 8 -or $DerSig[0] -ne 0x30) {
    throw "Invalid ASN.1 DER ECDSA signature header"
  }
  $idx = 2
  if (($DerSig[1] -band 0x80) -ne 0) {
    $lenBytes = $DerSig[1] -band 0x7F
    $idx = 2 + $lenBytes
  }
  if ($DerSig[$idx] -ne 0x02) { throw "Invalid ASN.1 INTEGER tag for R" }
  $rLen = [int]$DerSig[$idx + 1]
  $rBytes = $DerSig[($idx + 2)..($idx + 1 + $rLen)]
  $idx = $idx + 2 + $rLen
  if ($DerSig[$idx] -ne 0x02) { throw "Invalid ASN.1 INTEGER tag for S" }
  $sLen = [int]$DerSig[$idx + 1]
  $sBytes = $DerSig[($idx + 2)..($idx + 1 + $sLen)]

  $out = New-Object byte[] 64
  $rTrim = if ($rBytes.Length -gt 32) { $rBytes[($rBytes.Length - 32)..($rBytes.Length - 1)] } else { $rBytes }
  $sTrim = if ($sBytes.Length -gt 32) { $sBytes[($sBytes.Length - 32)..($sBytes.Length - 1)] } else { $sBytes }
  [Array]::Copy($rTrim, 0, $out, 32 - $rTrim.Length, $rTrim.Length)
  [Array]::Copy($sTrim, 0, $out, 64 - $sTrim.Length, $sTrim.Length)
  return $out
}

function Verify-CarbonReleaseSignature([string]$PemText, [byte[]]$DataBytes, [byte[]]$DerSig) {
  $B64 = ($PemText -split "`r?`n" | Where-Object { $_ -and ($_ -notmatch "^-----") }) -join ""
  $Spki = [Convert]::FromBase64String($B64)
  if ($Spki.Length -ne 91 -or $Spki[26] -ne 0x04) {
    throw "Unsupported SPKI P-256 public key DER length ($($Spki.Length))"
  }
  $X = $Spki[27..58]
  $Y = $Spki[59..90]
  $Params = New-Object System.Security.Cryptography.ECParameters
  $Params.Curve = [System.Security.Cryptography.ECCurve+NamedCurves]::nistP256
  $Params.Q = New-Object System.Security.Cryptography.ECPoint
  $Params.Q.X = $X
  $Params.Q.Y = $Y
  $Ecdsa = [System.Security.Cryptography.ECDsa]::Create($Params)
  try {
    $IeeeSig = Convert-Asn1DerEcdsaToP1363 -DerSig $DerSig
    if (-not $Ecdsa.VerifyData($DataBytes, $IeeeSig, [System.Security.Cryptography.HashAlgorithmName]::SHA256)) {
      throw "ECDSA-P256 signature verification FAILED for SHA256SUMS.txt"
    }
  } finally {
    $Ecdsa.Dispose()
  }
}

try {
  if (-not [string]::IsNullOrWhiteSpace($AllowTestPubkey)) {
    Write-CarbonLog "WARNING: using test public key override ($AllowTestPubkey) — NOT for production installs"
    $ActivePem = [System.IO.File]::ReadAllText($AllowTestPubkey)
  } else {
    $PemBytes = [System.Text.Encoding]::ASCII.GetBytes($EmbeddedSigningPubPem)
    $Sha256Alg = [System.Security.Cryptography.SHA256]::Create()
    try {
      $GotPubHash = ([BitConverter]::ToString($Sha256Alg.ComputeHash($PemBytes)).Replace("-", "")).ToLowerInvariant()
    } finally {
      $Sha256Alg.Dispose()
    }
    if ($GotPubHash -ne $SigningPubFingerprint) {
      throw "Embedded release public key fingerprint mismatch ($GotPubHash)"
    }
    $ActivePem = $EmbeddedSigningPubPem
  }

  Write-CarbonLog "Fetching signed Carbon release manifest '$Version'..."
  $ManifestPath = Join-Path $StageDir "SHA256SUMS.txt"
  $SigPath = Join-Path $StageDir "SHA256SUMS.txt.sig"
  Fetch-CarbonAsset "SHA256SUMS.txt" $ManifestPath
  Fetch-CarbonAsset "SHA256SUMS.txt.sig" $SigPath

  $ManifestBytes = [System.IO.File]::ReadAllBytes($ManifestPath)
  $SigB64 = ([System.IO.File]::ReadAllText($SigPath) -replace "\s+", "")
  $SigDer = [Convert]::FromBase64String($SigB64)
  Write-CarbonLog "Verifying ECDSA-P256 release signature..."
  Verify-CarbonReleaseSignature -PemText $ActivePem -DataBytes $ManifestBytes -DerSig $SigDer

  $ManifestMap = @{}
  foreach ($Line in [System.IO.File]::ReadAllLines($ManifestPath)) {
    $Trimmed = $Line.Trim()
    if ([string]::IsNullOrWhiteSpace($Trimmed) -or $Trimmed.StartsWith("#")) { continue }
    $Parts = $Trimmed -split "\s+", 2
    if ($Parts.Count -eq 2) {
      $ManifestMap[$Parts[1].TrimStart("*")] = $Parts[0].ToLowerInvariant()
    }
  }

  $ZipAsset = "$BaseAsset.zip"
  $RawAsset = "$BaseAsset.exe"
  if ($ManifestMap.ContainsKey($ZipAsset)) {
    $DownloadAsset = $ZipAsset
    $IsZip = $true
  } elseif ($ManifestMap.ContainsKey($RawAsset)) {
    $DownloadAsset = $RawAsset
    $IsZip = $false
  } else {
    throw "Neither $ZipAsset nor $RawAsset found in signed SHA256SUMS.txt"
  }

  $CloudsStage = Join-Path $StageDir "carbon-clouds.json"
  $PayloadStage = Join-Path $StageDir $DownloadAsset
  Fetch-CarbonAsset "carbon-clouds.json" $CloudsStage
  Fetch-CarbonAsset $DownloadAsset $PayloadStage

  foreach ($Asset in @($DownloadAsset, "carbon-clouds.json")) {
    if (-not $ManifestMap.ContainsKey($Asset)) {
      throw "$Asset missing from signed SHA256SUMS.txt"
    }
    $Expected = $ManifestMap[$Asset]
    $Actual = Get-FileSha256Lower (Join-Path $StageDir $Asset)
    if ($Expected -ne $Actual) {
      throw "SHA-256 mismatch for $Asset (expected $Expected, got $Actual)"
    }
  }

  if ($IsZip) {
    $ExtractDir = Join-Path $StageDir "extracted"
    Expand-Archive -LiteralPath $PayloadStage -DestinationPath $ExtractDir -Force
    $ExeSource = Join-Path $ExtractDir "carbon.exe"
    if (-not (Test-Path -LiteralPath $ExeSource)) {
      throw "Archive $DownloadAsset did not contain carbon.exe"
    }
    if ($ManifestMap.ContainsKey($RawAsset)) {
      $ActualExeSha = Get-FileSha256Lower $ExeSource
      if ($ActualExeSha -ne $ManifestMap[$RawAsset]) {
        throw "Extracted carbon.exe SHA-256 mismatch against $RawAsset"
      }
    }
  } else {
    $ExeSource = $PayloadStage
  }

  New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
  New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null

  $DestExe = Join-Path $BinDir "carbon.exe"
  Copy-Item -LiteralPath $ExeSource -Destination $DestExe -Force
  Copy-Item -LiteralPath $CloudsStage -Destination (Join-Path $ConfigDir "carbon-clouds.json") -Force
  Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $ConfigDir "SHA256SUMS.txt") -Force
  Copy-Item -LiteralPath $SigPath -Destination (Join-Path $ConfigDir "SHA256SUMS.txt.sig") -Force

  Write-CarbonLog "Installed verified binary     -> $DestExe"
  Write-CarbonLog "Installed verified cloud pins -> $(Join-Path $ConfigDir 'carbon-clouds.json')"

  $env:CARBON_CONFIG_DIR = $ConfigDir
  & $DestExe version | Write-Host
  if (-not $SkipPing) {
    & $DestExe ping --cloud prod | Write-Host
  }
  & $DestExe self-verify | Out-Null
  Write-CarbonLog "Ready! Try: carbon quickstart  |  carbon data.csv  |  carbon self-verify"
} finally {
  if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse -Force -ErrorAction SilentlyContinue
  }
}
