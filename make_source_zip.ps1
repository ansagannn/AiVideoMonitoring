$src = 'C:\Users\Admin\Desktop\AiMoniitoringv5'
$zip = 'C:\Users\Admin\Desktop\AiMoniitoringv5-source.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
$files = Get-ChildItem -Path $src -Recurse -File | Where-Object {
    $p = $_.FullName
    $p -notlike '*\\.git\\*' -and
    $p -notlike '*\\.vscode\\*' -and
    $p -notlike '*\\.venv\\*' -and
    $p -notlike '*\\frontend\\node_modules\\*' -and
    $p -notlike '*\\backend\\monitoring.db'
}
Compress-Archive -Path $files -DestinationPath $zip -Force
Write-Output "ZIP_SIZE_MB: $([math]::Round((Get-Item $zip).Length / 1MB, 2))"