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
  [string]$Version = "carbon-cli-v1.3.0",
  [string]$BinDir = "",
  [string]$ConfigDir = "",
  [string]$BaseUrl = "",
  [string]$SourceDir = "",
  [string]$AllowTestPubkey = "",
  [switch]$SkipPing,
  [switch]$NoModifyPath,
  [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

try {
  $TlsProto = [Net.SecurityProtocolType]::Tls12
  if ([Enum]::IsDefined([Net.SecurityProtocolType], "Tls13")) {
    $TlsProto = $TlsProto -bor [Net.SecurityProtocolType]::Tls13
  }
  [Net.ServicePointManager]::SecurityProtocol = $TlsProto
} catch {}
$ProgressPreference = "SilentlyContinue"

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

$IsWinHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

if ($Uninstall) {
  $DestExe = Join-Path $BinDir "carbon.exe"
  if (Test-Path -LiteralPath $DestExe) {
    Remove-Item -LiteralPath $DestExe -Force -ErrorAction SilentlyContinue
    Write-CarbonLog "Removed $DestExe"
  }
  Get-ChildItem -LiteralPath $BinDir -Filter "carbon.exe.old.*" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
  }
  foreach ($CfgFile in @("carbon-clouds.json", "SHA256SUMS.txt", "SHA256SUMS.txt.sig")) {
    $TargetCfg = Join-Path $ConfigDir $CfgFile
    if (Test-Path -LiteralPath $TargetCfg) {
      Remove-Item -LiteralPath $TargetCfg -Force -ErrorAction SilentlyContinue
      Write-CarbonLog "Removed $TargetCfg"
    }
  }
  if ($IsWinHost -and -not $NoModifyPath) {
    $CurrentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not [string]::IsNullOrWhiteSpace($CurrentUserPath)) {
      $Filtered = ($CurrentUserPath -split ";" | Where-Object {
        $_ -and ($_.TrimEnd("\") -ne $BinDir.TrimEnd("\"))
      }) -join ";"
      if ($Filtered -ne $CurrentUserPath) {
        [Environment]::SetEnvironmentVariable("Path", $Filtered, "User")
        Write-CarbonLog "Removed $BinDir from User Path"
      }
    }
  }
  Write-CarbonLog "Uninstall complete (saved zero-knowledge run keys under $ConfigDir\runs were preserved)."
  return
}

function Resolve-NativeWindowsArch {
  # Detect true native machine architecture even when PowerShell 5.1 runs under x64 WOW64 on Windows 11 ARM64.
  if ($IsWinHost) {
    try {
      $RegArch = (Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" -Name "PROCESSOR_ARCHITECTURE" -ErrorAction SilentlyContinue).PROCESSOR_ARCHITECTURE
      if (-not [string]::IsNullOrWhiteSpace($RegArch)) {
        return $RegArch.Trim().ToLowerInvariant()
      }
    } catch {}
  }
  if (-not [string]::IsNullOrWhiteSpace($env:PROCESSOR_ARCHITEW6432)) {
    return $env:PROCESSOR_ARCHITEW6432.Trim().ToLowerInvariant()
  }
  if (-not [string]::IsNullOrWhiteSpace($env:PROCESSOR_ARCHITECTURE)) {
    return $env:PROCESSOR_ARCHITECTURE.Trim().ToLowerInvariant()
  }
  return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
}

$ArchRaw = Resolve-NativeWindowsArch
switch ($ArchRaw) {
  "x64"   { $BaseAsset = "carbon-windows-x86_64-msvc"; $ArchSlug = "x86_64" }
  "amd64" { $BaseAsset = "carbon-windows-x86_64-msvc"; $ArchSlug = "x86_64" }
  "arm64" { $BaseAsset = "carbon-windows-aarch64-msvc"; $ArchSlug = "aarch64" }
  default { throw "[carbon-install] ERROR: Unsupported Windows CPU architecture '$ArchRaw' (expected x64/AMD64 or ARM64)" }
}

$TelemetryOptOut = ($env:DO_NOT_TRACK -eq "1" -or $env:CARBON_NO_TELEMETRY -eq "1")
$InstallerUserAgent = if ($TelemetryOptOut) {
  "jope-carbon-installer/1"
} else {
  "jope-carbon-installer/1 ($Version; windows-$ArchSlug)"
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  if ($TelemetryOptOut) {
    $EffectiveBaseUrl = "https://github.com/neverhuman/JopeDime/releases/download/$Version"
    $MirrorUrl = "https://downloads.neverhuman.org/carbon/releases/$Version"
  } else {
    $EffectiveBaseUrl = "https://downloads.neverhuman.org/carbon/releases/$Version"
    $MirrorUrl = "https://github.com/neverhuman/JopeDime/releases/download/$Version"
  }
  $UseSecondaryMirror = $true
} else {
  $EffectiveBaseUrl = $BaseUrl.TrimEnd("/")
  if (-not ($EffectiveBaseUrl -match '^https://' -or $EffectiveBaseUrl -match '^http://(127\.0\.0\.1|localhost|\[::1\])(:|/|$)')) {
    Write-CarbonLog "ERROR: Insecure or unsupported -BaseUrl '$EffectiveBaseUrl' (HTTPS required for non-loopback hosts)"
    exit 1
  }
  $MirrorUrl = "https://github.com/neverhuman/JopeDime/releases/download/$Version"
  $UseSecondaryMirror = $false
}

$StageParent = if (-not [string]::IsNullOrWhiteSpace($env:CARBON_INSTALL_STAGE_ROOT)) {
  $env:CARBON_INSTALL_STAGE_ROOT
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA "Temp"
} else {
  [System.IO.Path]::GetTempPath()
}
New-Item -ItemType Directory -Path $StageParent -Force | Out-Null
$StageDir = Join-Path $StageParent ("carbon-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $StageDir | Out-Null

function Get-FileSha256Lower([string]$Path) {
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
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
    Invoke-WebRequest -Uri $Primary -OutFile $DestPath -UseBasicParsing -UserAgent $InstallerUserAgent -TimeoutSec 30
  } catch {
    if (-not $UseSecondaryMirror) { throw }
    Write-CarbonLog "Primary download unavailable for $AssetName; failing over to $MirrorUrl..."
    Invoke-WebRequest -Uri "$MirrorUrl/$AssetName" -OutFile $DestPath -UseBasicParsing -UserAgent $InstallerUserAgent -TimeoutSec 30
  }
}

function Convert-Asn1DerEcdsaToP1363([byte[]]$DerSig) {
  if ($null -eq $DerSig -or $DerSig.Length -lt 8 -or $DerSig[0] -ne 0x30) {
    throw "Invalid ASN.1 DER ECDSA signature header"
  }
  $idx = 2
  $seqLen = 0
  if (($DerSig[1] -band 0x80) -eq 0) {
    $seqLen = [int]$DerSig[1]
    $idx = 2
  } else {
    $lenBytes = [int]($DerSig[1] -band 0x7F)
    if ($lenBytes -ne 1 -or $DerSig.Length -lt 9) {
      throw "Invalid ASN.1 DER ECDSA sequence length encoding"
    }
    $seqLen = [int]$DerSig[2]
    $idx = 3
  }
  if (($idx + $seqLen) -ne $DerSig.Length) {
    throw "ASN.1 DER ECDSA signature has trailing or truncated outer bytes"
  }

  if (($idx + 2) -gt $DerSig.Length -or $DerSig[$idx] -ne 0x02) {
    throw "Invalid ASN.1 INTEGER tag for R"
  }
  $rLen = [int]$DerSig[$idx + 1]
  if ($rLen -lt 1 -or $rLen -gt 33 -or ($idx + 2 + $rLen) -ge $DerSig.Length) {
    throw "Invalid ASN.1 INTEGER length for R ($rLen)"
  }
  $rOffset = $idx + 2
  $rCopyLen = $rLen
  if ($rCopyLen -eq 33) {
    if ($DerSig[$rOffset] -ne 0x00) {
      throw "Overlong non-padded ASN.1 INTEGER for R"
    }
    $rOffset += 1
    $rCopyLen = 32
  }

  $idx = $idx + 2 + $rLen
  if (($idx + 2) -gt $DerSig.Length -or $DerSig[$idx] -ne 0x02) {
    throw "Invalid ASN.1 INTEGER tag for S"
  }
  $sLen = [int]$DerSig[$idx + 1]
  if ($sLen -lt 1 -or $sLen -gt 33 -or ($idx + 2 + $sLen) -ne $DerSig.Length) {
    throw "Invalid ASN.1 INTEGER length for S ($sLen)"
  }
  $sOffset = $idx + 2
  $sCopyLen = $sLen
  if ($sCopyLen -eq 33) {
    if ($DerSig[$sOffset] -ne 0x00) {
      throw "Overlong non-padded ASN.1 INTEGER for S"
    }
    $sOffset += 1
    $sCopyLen = 32
  }

  $out = New-Object byte[] 64
  [System.Buffer]::BlockCopy($DerSig, $rOffset, $out, 32 - $rCopyLen, $rCopyLen)
  [System.Buffer]::BlockCopy($DerSig, $sOffset, $out, 64 - $sCopyLen, $sCopyLen)
  return ,$out
}

function Verify-CarbonReleaseSignature([string]$PemText, [byte[]]$DataBytes, [byte[]]$DerSig) {
  $B64 = ($PemText -split "`r?`n" | Where-Object { $_ -and ($_ -notmatch "^-----") }) -join ""
  $Spki = [Convert]::FromBase64String($B64)
  if ($Spki.Length -ne 91 -or $Spki[26] -ne 0x04) {
    throw "Unsupported SPKI P-256 public key DER length ($($Spki.Length))"
  }
  $X = New-Object byte[] 32
  $Y = New-Object byte[] 32
  [System.Buffer]::BlockCopy($Spki, 27, $X, 0, 32)
  [System.Buffer]::BlockCopy($Spki, 59, $Y, 0, 32)
  $Pt = New-Object System.Security.Cryptography.ECPoint
  $Pt.X = $X
  $Pt.Y = $Y
  $Params = New-Object System.Security.Cryptography.ECParameters
  $Params.Curve = [System.Security.Cryptography.ECCurve]::CreateFromValue("1.2.840.10045.3.1.7")
  $Params.Q = $Pt
  $Ecdsa = [System.Security.Cryptography.ECDsa]::Create($Params)
  try {
    [byte[]]$IeeeSig = Convert-Asn1DerEcdsaToP1363 -DerSig $DerSig
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
    $ActivePem = [System.IO.File]::ReadAllText($AllowTestPubkey).Replace("`r`n", "`n")
  } else {
    $NormalizedEmbeddedPem = $EmbeddedSigningPubPem.Replace("`r`n", "`n")
    $PemBytes = [System.Text.Encoding]::ASCII.GetBytes($NormalizedEmbeddedPem)
    $Sha256Alg = [System.Security.Cryptography.SHA256]::Create()
    try {
      $GotPubHash = ([BitConverter]::ToString($Sha256Alg.ComputeHash($PemBytes)).Replace("-", "")).ToLowerInvariant()
    } finally {
      $Sha256Alg.Dispose()
    }
    if ($GotPubHash -ne $SigningPubFingerprint) {
      throw "Embedded release public key fingerprint mismatch ($GotPubHash)"
    }
    $ActivePem = $NormalizedEmbeddedPem
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

  Get-ChildItem -LiteralPath $BinDir -Filter "carbon.exe.old.*" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
  }

  $DestExe = Join-Path $BinDir "carbon.exe"
  $DestNew = Join-Path $BinDir ("carbon.exe.new." + $PID)
  Copy-Item -LiteralPath $ExeSource -Destination $DestNew -Force
  try {
    Move-Item -LiteralPath $DestNew -Destination $DestExe -Force
  } catch {
    $OldLocked = "$DestExe.old.$PID"
    if (Test-Path -LiteralPath $DestExe) {
      Move-Item -LiteralPath $DestExe -Destination $OldLocked -Force
    }
    Move-Item -LiteralPath $DestNew -Destination $DestExe -Force
  }

  Copy-Item -LiteralPath $CloudsStage -Destination (Join-Path $ConfigDir "carbon-clouds.json") -Force
  Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $ConfigDir "SHA256SUMS.txt") -Force
  Copy-Item -LiteralPath $SigPath -Destination (Join-Path $ConfigDir "SHA256SUMS.txt.sig") -Force

  Write-CarbonLog "Installed verified binary     -> $DestExe"
  Write-CarbonLog "Installed verified cloud pins -> $(Join-Path $ConfigDir 'carbon-clouds.json')"

  $env:CARBON_CONFIG_DIR = $ConfigDir
  $VersionOut = & $DestExe version
  $VersionCode = $LASTEXITCODE
  if ($null -ne $VersionOut) { $VersionOut | Write-Host }
  if ($VersionCode -ne 0) {
    throw "Installed carbon.exe failed 'version' check (exit code $VersionCode)"
  }
  if (-not [string]::IsNullOrWhiteSpace($SourceDir)) {
    $SkipPing = $true
  }
  if (-not $SkipPing) {
    try {
      $PingOut = & $DestExe ping --cloud prod
      $PingCode = $LASTEXITCODE
      if ($null -ne $PingOut) { $PingOut | Write-Host }
      if ($PingCode -ne 0) {
        Write-CarbonLog "WARNING: Post-install 'carbon ping --cloud prod' returned exit code $PingCode (binary and cryptographic pins are verified; outbound HTTPS may be offline or restricted)."
      }
    } catch {
      Write-CarbonLog "WARNING: Post-install 'carbon ping --cloud prod' failed ($($_.Exception.Message)); binary and cryptographic pins are verified."
    }
  }
  if ([string]::IsNullOrWhiteSpace($AllowTestPubkey)) {
    $null = & $DestExe self-verify
    $VerifyCode = $LASTEXITCODE
    if ($VerifyCode -ne 0) {
      throw "Post-install 'carbon self-verify' failed (exit code $VerifyCode)"
    }
  }

  $PathSeparator = [System.IO.Path]::PathSeparator
  $SessionPaths = ($env:PATH -split [regex]::Escape([string]$PathSeparator)) | Where-Object { $_ }
  $NormalizedBin = $BinDir.TrimEnd("\").TrimEnd("/")
  $InSessionPath = $false
  foreach ($Entry in $SessionPaths) {
    if ($Entry.TrimEnd("\").TrimEnd("/") -ieq $NormalizedBin) {
      $InSessionPath = $true
      break
    }
  }
  if (-not $InSessionPath) {
    $env:PATH = "$BinDir$PathSeparator$($env:PATH)"
  }

  if ($IsWinHost -and -not $NoModifyPath -and [string]::IsNullOrWhiteSpace($AllowTestPubkey)) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $UserEntries = if ([string]::IsNullOrWhiteSpace($UserPath)) { @() } else { $UserPath -split ";" | Where-Object { $_ } }
    $InUserPath = $false
    foreach ($Entry in $UserEntries) {
      if ($Entry.TrimEnd("\") -ieq $NormalizedBin) {
        $InUserPath = $true
        break
      }
    }
    if (-not $InUserPath) {
      $UpdatedUserPath = if ([string]::IsNullOrWhiteSpace($UserPath)) { $BinDir } else { "$BinDir;$UserPath" }
      [Environment]::SetEnvironmentVariable("Path", $UpdatedUserPath, "User")
      Write-CarbonLog "Added $BinDir to persistent Windows User Path"
    }
  }

  Write-CarbonLog "Ready! Try: carbon quickstart  |  carbon data.csv  |  carbon self-verify"
} catch {
  Write-CarbonLog "ERROR: $($_.Exception.Message)"
  if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  exit 1
} finally {
  if (Test-Path -LiteralPath $StageDir) {
    Remove-Item -LiteralPath $StageDir -Recurse -Force -ErrorAction SilentlyContinue
  }
}
