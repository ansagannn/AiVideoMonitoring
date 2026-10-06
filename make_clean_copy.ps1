$src = 'C:/Users/Admin/Desktop/AiMoniitoringv5'
$dst = 'C:/Users/Admin/Desktop/AiMoniitoringv5-clean'
if(Test-Path $dst){ Remove-Item $dst -Recurse -Force }
$excludeDirs = @('.git', '.vscode', '.venv', 'backend\\.venv', 'frontend\\node_modules')
$excludeFiles = @('backend\\monitoring.db')
Get-ChildItem -Path $src -Recurse -Force | ForEach-Object {
    if ($_.PSIsContainer) { return }
    foreach ($ex in $excludeFiles) {
        if ($_.FullName -like "*$ex") { return }
    }
    foreach ($ex in $excludeDirs) {
        if ($_.FullName -like "*$ex*") { return }
    }
    $rel = $_.FullName.Substring($src.Length + 1)
    $target = Join-Path $dst $rel
    $parent = Split-Path $target
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Copy-Item -Path $_.FullName -Destination $target -Force
}
Write-Output 'CLEAN_COPY_CREATED'
Write-Output ((Get-ChildItem -Path $dst -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)