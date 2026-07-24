# Supervisor del entrenamiento: relanza train_gen.py --resume cada vez que muere.
#
# El equipo hace BSOD 0x00020001 (HYPERVISOR_ERROR) bajo carga de GPU sostenida --
# 22/07/2026 a las 10:57 y a las 11:01, con el entrenamiento en marcha. Es un fallo
# de driver/VBS, no del codigo, y desde Python no se arregla. Lo que SI se puede es
# hacer que no cueste nada: train_gen.py guarda ckpt/gen_last.pt al final de CADA
# epoca, asi que como mucho se pierde la epoca en curso (~1 min).
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File vigilar.ps1
#   powershell -ExecutionPolicy Bypass -File vigilar.ps1 -Epochs 800 -Bs 4
#
# Para que sobreviva tambien al reinicio, registrar como tarea al iniciar sesion:
#   .\vigilar.ps1 -Instalar
param(
    [int]$Epochs = 1000,
    [int]$Bs = 3,
    [int]$Patch = 96,
    [int]$Mon = 10,
    [double]$VesselW = 12.0,
    [double]$Lr = 3e-4,
    [string]$From = "gen_best.pt",   # checkpoint del PRIMER arranque
    [int]$DesdeEp = 799,             # epoca del checkpoint inicial (para decidir en los reinicios)
    [switch]$Instalar
)

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "python"

if ($Instalar) {
    # Este equipo esta gestionado: `schtasks /Create` da "Acceso denegado" incluso
    # sin /RL HIGHEST (probado). La carpeta de Inicio del usuario no necesita
    # ningun privilegio y consigue lo mismo: arrancar al iniciar sesion.
    $inicio = [Environment]::GetFolderPath('Startup')
    $lnk = Join-Path $inicio "cta_gen_train.cmd"
    $cmd = "@echo off`r`n" +
           "powershell -ExecutionPolicy Bypass -WindowStyle Minimized -File " +
           "`"$raiz\vigilar.ps1`" -Epochs $Epochs -Bs $Bs -Patch $Patch -Mon $Mon " +
           "-VesselW $VesselW -Lr $Lr -From $From -DesdeEp $DesdeEp`r`n"
    Set-Content -Path $lnk -Value $cmd -Encoding ASCII
    Write-Output "instalado en la carpeta de Inicio: $lnk"
    Write-Output "el entrenamiento se reanuda solo al iniciar sesion tras un reinicio."
    Write-Output "para quitarlo:  Remove-Item `"$lnk`""
    exit
}

# Best-effort: bajar el limite de potencia de la GPU reduce el pico que dispara el
# BSOD. Requiere admin; si falla, no pasa nada y se sigue igual.
try { & nvidia-smi -pl 250 2>$null | Out-Null } catch { }

$log = Join-Path $raiz "vigilar.log"
$intento = 0
while ($true) {
    $intento++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $log "=== intento $intento  $ts ==="
    Write-Output "=== intento $intento  $ts ==="

    # De que checkpoint reanudar. El primer arranque parte de $From (gen_best.pt,
    # sin optimizador -> el codigo reinicia los momentos y aplica la rampa de LR).
    # Pero en cuanto el fine-tune graba un gen_last.pt con epoca > la inicial, hay
    # que seguir por ESE, no volver a gen_best: si no, un reinicio a media corrida
    # perderia todo el progreso y volveria a reiniciar los momentos. Como un
    # reinicio del equipo arranca una vigilar.ps1 nueva ($intento=1), la decision no
    # puede depender de $intento: se mira la epoca real de gen_last.pt.
    $last = Join-Path $raiz "ckpt\gen_last.pt"
    $desde = $From
    if (Test-Path $last) {
        $epLast = & $py -c "import torch;print(torch.load(r'$last',map_location='cpu',weights_only=False).get('epoch',-1))"
        if ([int]$epLast -gt $DesdeEp) {
            $desde = "gen_last.pt"
            Add-Content $log "gen_last.pt va por la epoca $epLast (> $DesdeEp): se reanuda de ahi"
        }
    }

    & $py (Join-Path $raiz "train_gen.py") --resume --from $desde --epochs $Epochs `
        --bs $Bs --patch $Patch --mon $Mon --vessel_w $VesselW --lr $Lr `
        2>&1 | Tee-Object -FilePath $log -Append
    $rc = $LASTEXITCODE

    if ($rc -eq 0) {
        Add-Content $log "entrenamiento terminado limpiamente (rc=0)"
        Write-Output "entrenamiento terminado limpiamente."
        break
    }

    # rc != 0: o el proceso murio (CUDA, OOM) o el equipo se reinicio y esto arranca
    # de cero. En ambos casos se reanuda desde gen_last.pt. Pausa para que la GPU
    # se enfrie y el driver se recupere antes de volver a cargarla.
    Add-Content $log "murio con rc=$rc -- reanudando desde ckpt/gen_last.pt en 60 s"
    Write-Output "murio con rc=$rc -- reanudando en 60 s"
    Start-Sleep -Seconds 60
}
