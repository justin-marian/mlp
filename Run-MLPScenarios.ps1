<# 
.SYNOPSIS
  Run MLP training/eval scenarios for classification (Iris) and regression (Diabetes).

.DESCRIPTION
  This script calls:
    python main.py -t classification -d iris ...
    python main.py -t regression      -d diabetes ...

  It mirrors the argument patterns from your argparse config and includes
  multiple curated scenarios per task.

.USAGE
  # From project root (where main.py exists)
  # Run only classification scenarios:
  .\Run-MLPScenarios.ps1 -Only Classification

  # Run only regression scenarios:
  .\Run-MLPScenarios.ps1 -Only Regression

  # Run both:
  .\Run-MLPScenarios.ps1

.PARAMETER PythonExe
  Path or command name for Python. Defaults to "python".
#>

param(
  [string]$PythonExe = "python",
  [ValidateSet("All","Classification","Regression")]
  [string]$Only = "All"
)

function Write-Header($text) {
  Write-Host ""
  Write-Host ("=" * 80) -ForegroundColor DarkGray
  Write-Host ">> $text" -ForegroundColor Cyan
  Write-Host ("=" * 80) -ForegroundColor DarkGray
}

function Invoke-Scenario {
  param(
    [string]$Name,
    [string[]]$ScenarioArgs
  )
  Write-Host "`n--- Scenario: $Name ---" -ForegroundColor Yellow
  Write-Host "Args: $($ScenarioArgs -join ' ')" -ForegroundColor DarkGray
  & $PythonExe @ScenarioArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Scenario '$Name' exited with code $LASTEXITCODE."
  }
}

function New-ArgumentList {
  param(
    [ValidateSet("classification","regression")][string]$Task,
    [ValidateSet("iris","diabetes")][string]$Dataset,
    [int[]]$Layers,
    [ValidateSet("relu","sigmoid","tanh")][string]$Activation = "relu",
    [ValidateSet("softmax","linear")][string]$Output = "softmax",
    [double]$LearningRate = 0.01,
    [double]$DecayRate = 0.001,
    [int]$Epochs = 100,
    [double]$TestSize = 0.2,
    [int]$Seed = 42,
    [switch]$ShowLoss,
    [ValidateSet("he","xavier","random")][string]$WeightInit = "he",
    [double]$L1 = 0.0,
    [double]$L2 = 0.0,
    [string]$ExtensionName = "",
    [int[]]$WantedFeatures = @()
  )
  $argList = @(
    "main.py",
    "-t", $Task,
    "-d", $Dataset,
    "-a", $Activation,
    "-o", $Output,
    "-lr", $LearningRate,
    "--decay_rate", $DecayRate,
    "-e", $Epochs,
    "-ts", $TestSize,
    "-s", $Seed,
    "-wi", $WeightInit,
    "-l"
  )
  $argList += $Layers

  # Regularization (optional)
  if ($L1 -gt 0) { $argList += @("--l1", $L1) }
  if ($L2 -gt 0) { $argList += @("--l2", $L2) }

  # Optional extras
  if ($ExtensionName -ne "") { $argList += @("--extension_name", $ExtensionName) }
  if ($WantedFeatures.Count -gt 0) { $argList += @("--wanted_features"); $argList += $WantedFeatures }

  if ($ShowLoss) { $argList += "--show_loss" }
  return ,$argList
}

function Invoke-ClassificationScenarios {
  Write-Header "Classification (Iris) Scenarios"

  # 1) Fast smoke
  Invoke-Scenario "Clf: Fast Smoke (relu, small net)" (New-ArgumentList `
    -Task classification -Dataset iris `
    -Layers @(16, 3) `
    -Activation relu `
    -LearningRate 0.05 `
    -Epochs 60 `
    -TestSize 0.25 `
    -Seed 7 `
    -WeightInit he `
    -DecayRate 0.001 `
    -Output softmax `
    -ExtensionName "iris_smoke" `
    -WantedFeatures @(0,1) `
    -ShowLoss)

  # 2) Strong baseline
  Invoke-Scenario "Clf: Strong Baseline (relu, deeper)" (New-ArgumentList `
    -Task classification -Dataset iris `
    -Layers @(32, 16, 3) `
    -Activation relu `
    -LearningRate 0.05 `
    -Epochs 200 `
    -TestSize 0.2 `
    -Seed 42 `
    -WeightInit he `
    -DecayRate 0.001 `
    -Output softmax `
    -ExtensionName "iris_baseline" `
    -WantedFeatures @(2,3) `
    -ShowLoss)

  # 3) Tanh + Xavier
  Invoke-Scenario "Clf: Tanh + Xavier" (New-ArgumentList `
    -Task classification -Dataset iris `
    -Layers @(64, 16, 3) `
    -Activation tanh `
    -LearningRate 0.01 `
    -Epochs 250 `
    -TestSize 0.2 `
    -Seed 123 `
    -WeightInit xavier `
    -DecayRate 0.002 `
    -Output softmax `
    -ExtensionName "iris_tanh" `
    -ShowLoss)

  # 4) Sigmoid + Xavier
  Invoke-Scenario "Clf: Sigmoid + Xavier" (New-ArgumentList `
    -Task classification -Dataset iris `
    -Layers @(16, 3) `
    -Activation sigmoid `
    -LearningRate 0.02 `
    -Epochs 150 `
    -TestSize 0.3 `
    -Seed 99 `
    -WeightInit xavier `
    -DecayRate 0.001 `
    -Output softmax `
    -ExtensionName "iris_sigmoid" `
    -ShowLoss)

  # 5) Wide shallow + small L2 (for stability)
  Invoke-Scenario "Clf: Wide Shallow (relu, small L2)" (New-ArgumentList `
    -Task classification -Dataset iris `
    -Layers @(128, 3) `
    -Activation relu `
    -LearningRate 0.03 `
    -Epochs 80 `
    -TestSize 0.2 `
    -Seed 2025 `
    -WeightInit he `
    -DecayRate 0.0005 `
    -L2 0.0005 `
    -ExtensionName "iris_wide")
}

function Invoke-RegressionScenarios {
  Write-Header "Regression (Diabetes) Scenarios"

  # 1) Very simple linear baseline (1 hidden layer, strong ridge, slow LR)
  Invoke-Scenario "Reg: Linear Baseline (ridge strong)" (New-ArgumentList `
    -Task regression -Dataset diabetes `
    -Layers @(4, 1) `
    -Activation relu `
    -LearningRate 0.005 `
    -Epochs 1000 `
    -TestSize 0.2 `
    -Seed 42 `
    -WeightInit xavier `
    -DecayRate 0.0002 `
    -L2 0.02 `
    -ExtensionName "diab_linear_ridge_slow" `
  )

  # 2) Small net (plain MSE, conservative LR)
  Invoke-Scenario "Reg: Small MSE (relu, 16x1)" (New-ArgumentList `
    -Task regression -Dataset diabetes `
    -Layers @(16, 1) `
    -Activation relu `
    -LearningRate 0.1 `
    -Epochs 800 `
    -TestSize 0.25 `
    -Seed 7 `
    -WeightInit he `
    -DecayRate 0.0005 `
    -Output linear `
    -ExtensionName "diab_smoke" `
    -ShowLoss `
  )

  # 3) Strong baseline (tanh + Xavier, Ridge)
  Invoke-Scenario "Reg: Strong Baseline (tanh, Ridge)" (New-ArgumentList `
    -Task regression -Dataset diabetes `
    -Layers @(64, 32, 1) `
    -Activation tanh `
    -LearningRate 0.1 `
    -Epochs 800 `
    -TestSize 0.2 `
    -Seed 42 `
    -WeightInit xavier `
    -DecayRate 0.0005 `
    -L2 0.003 `
    -Output linear `
    -ExtensionName "diab_ridge" `
    -ShowLoss `
  )

  # 4) Larger net (ReLU + He, Ridge)
  Invoke-Scenario "Reg: Larger Net (relu, he, Ridge)" (New-ArgumentList `
    -Task regression -Dataset diabetes `
    -Layers @(128, 64, 1) `
    -Activation relu `
    -LearningRate 0.1 `
    -Epochs 400 `
    -TestSize 0.2 `
    -Seed 11 `
    -WeightInit he `
    -DecayRate 0.0005 `
    -L2 0.001 `
    -Output linear `
    -ExtensionName "diab_large" `
  )

  # 4) Lasso regularization (medium net, conservative LR)
  Invoke-Scenario "Reg: Lasso (relu, xavier)" (New-ArgumentList `
    -Task regression -Dataset diabetes `
    -Layers @(64, 1) `
    -Activation relu `
    -LearningRate 0.7 `
    -Epochs 800 `
    -TestSize 0.25 `
    -Seed 123 `
    -WeightInit xavier `
    -DecayRate 0.0005 `
    -L1 0.0005 `
    -Output linear `
    -ExtensionName "diab_lasso" `
    -ShowLoss `
  )
}


function Invoke-AllScenarios {
  Invoke-ClassificationScenarios
  Invoke-RegressionScenarios
}

$onlyNorm = $Only.Trim().ToLowerInvariant()
Write-Host "Selected -Only: '$Only' (normalized: '$onlyNorm')" -ForegroundColor Cyan

switch ($onlyNorm) {
  'classification' {
    Write-Host "Running classification scenarios..." -ForegroundColor Green; 
    Invoke-ClassificationScenarios
  }
  'regression' {
    Write-Host "Running regression scenarios..." -ForegroundColor Green; 
    Invoke-RegressionScenarios
  }
  default {
    Write-Host "Running ALL scenarios..." -ForegroundColor Green; 
    Invoke-ClassificationScenarios
    Invoke-RegressionScenarios
  }
}


# Clean all up any cache files created during runs
Write-Host "`nCleaning up __pycache__ directories..." -ForegroundColor DarkGray
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
  Write-Host "Removing $($_.FullName)" -ForegroundColor DarkGray
  Remove-Item -Path $_.FullName -Recurse -Force
}
Write-Host "`nAll scenarios completed." -ForegroundColor Green
