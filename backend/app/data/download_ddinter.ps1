$ErrorActionPreference = "Stop"

# Downloads DDInter CSVs (risk levels) into this folder.
# Source page: https://ddinter.scbdd.com/download/
#
# Output files will be:
#   ddinter_code_A.csv, ddinter_code_B.csv, ddinter_code_D.csv, ddinter_code_H.csv,
#   ddinter_code_L.csv, ddinter_code_P.csv, ddinter_code_R.csv, ddinter_code_V.csv
#
# Usage:
#   From repo root:
#     powershell -NoProfile -ExecutionPolicy Bypass -File backend/app/data/download_ddinter.ps1
#   From backend/:
#     powershell -NoProfile -ExecutionPolicy Bypass -File app/data/download_ddinter.ps1

$pairs = @(
  @{ Code = "A"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_A.csv" },
  @{ Code = "B"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_B.csv" },
  @{ Code = "D"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_D.csv" },
  @{ Code = "H"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_H.csv" },
  @{ Code = "L"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_L.csv" },
  @{ Code = "P"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_P.csv" },
  @{ Code = "R"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_R.csv" },
  @{ Code = "V"; Url = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_V.csv" }
)

$outDir = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($p in $pairs) {
  $outFile = Join-Path $outDir ("ddinter_code_{0}.csv" -f $p.Code)
  Write-Host ("Downloading DDInter code {0} -> {1}" -f $p.Code, $outFile)
  Invoke-WebRequest -Uri $p.Url -OutFile $outFile
}

Write-Host "Done."

