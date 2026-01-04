param(
  [int]$Generations = 5,
  [double[]]$Ratios = @(0.25, 0.5, 1.0),

  [int]$TotalN = 10000,
  [int]$TrainPairsPerGen = 10000,
  [int]$TestPairs = 2000,
  [int]$AnchorPrompts = 300,
  [int]$MaxContextTurns = 4,

  [double]$ValFrac = 0.05,
  [double]$TestFrac = 0.05,

  [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",

  [int]$MaxNewTokens = 100,
  [double]$Temperature = 0.9,
  [double]$TopP = 0.95,
  [int]$GenBatchSize = 16,

  [double]$EpochsGen00 = 5.0,
  [double]$EpochsNext  = 4.0,
  [int]$Batch = 2,
  [int]$GradAccum = 16,
  [int]$MaxLength = 1024,
  [double]$LR = 0.0002,

  [int]$Seed = 42,

  # default relative-to-project-root if startswith '\'
  [string]$SourceJson = "\data\raw\dialogues_combined.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Prevent PS7 "native stderr => error record" behavior if available
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}

function Timestamp { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }
function NowIso { (Get-Date).ToString("o") }
function Write-Log([string]$msg) { Write-Host ("[{0}] {1}" -f (Timestamp), $msg) }
function RatioLabel([double]$r) { return ("r{0:D3}" -f [int]([Math]::Round($r * 100))) }

# Must run where collapse_pipeline.py lives
if (-not (Test-Path ".\collapse_pipeline.py")) {
  throw "Run this script from the folder that contains collapse_pipeline.py"
}

# normalize SourceJson
if ($SourceJson.StartsWith("\")) {
  $SourceJson = Join-Path (Get-Location) $SourceJson.TrimStart("\")
}
$SourceJson = [System.IO.Path]::GetFullPath($SourceJson)

if (-not (Test-Path $SourceJson)) {
  throw "SourceJson not found: $SourceJson"
}

# Directories
New-Item -ItemType Directory -Force -Path ".\data", ".\data\synth", ".\data\mixed", ".\runs", ".\metrics", ".\logs" | Out-Null
$LogPath = Join-Path ".\logs" ("run_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

# Transcript keeps tqdm visible and logs console output
Start-Transcript -Path $LogPath -Append | Out-Null

try {
  $py = (Get-Command python).Source
  Write-Log ("Python: {0}" -f $py)
  Write-Log ("SourceJson: {0}" -f $SourceJson)
  Write-Log ("Generations: {0}, Ratios: {1}" -f $Generations, ($Ratios -join ", "))
  Write-Log ("TrainPairsPerGen: {0}, TotalN per run: {1}" -f $TrainPairsPerGen, $TotalN)

  # -----------------------
  # PREPARE (reshuffle + shards)
  # -----------------------
  Write-Log "PREPARE: building real shards + prompts (reshuffled splits)"
  & $py -u .\collapse_pipeline.py prepare `
    --out data `
    --source_json $SourceJson `
    --seed $Seed `
    --generations $Generations `
    --train_pairs_per_gen $TrainPairsPerGen `
    --test_pairs $TestPairs `
    --anchor_prompts $AnchorPrompts `
    --max_context_turns $MaxContextTurns `
    --val_frac $ValFrac `
    --test_frac $TestFrac
  if ($LASTEXITCODE -ne 0) { throw "prepare failed" }

  # -----------------------
  # GEN00 baseline (real only)
  # -----------------------
  if (-not (Test-Path ".\runs\gen00_adapter")) {
    Write-Log "TRAIN gen00 (real only baseline)"
    & $py -u .\collapse_pipeline.py train `
      --model $Model `
      --train_jsonl .\data\real_train_gen00.jsonl `
      --out_adapter .\runs\gen00_adapter `
      --epochs $EpochsGen00 `
      --batch $Batch `
      --grad_accum $GradAccum `
      --max_length $MaxLength `
      --lr $LR `
      --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "train gen00 failed" }
  } else {
    Write-Log "runs\gen00_adapter exists -> skipping train gen00"
  }

  if (-not (Test-Path ".\metrics\gen00.json")) {
    Write-Log "EVAL gen00"
    & $py -u .\collapse_pipeline.py eval `
      --model $Model `
      --adapter .\runs\gen00_adapter `
      --real_test .\data\real_test.jsonl `
      --anchor_prompts .\data\anchor_prompts.jsonl `
      --out_metrics .\metrics\gen00.json `
      --max_test_examples 800 `
      --anchor_limit 200 `
      --max_new_tokens $MaxNewTokens `
      --temperature $Temperature `
      --top_p $TopP `
      --max_length $MaxLength `
      --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "eval gen00 failed" }
  }

  # -----------------------
  # Generations 1..G and ratios
  # Each generation uses NEW real shard: real_train_genXX.jsonl
  # Synthetic is generated from previous model on prompts_genXX.jsonl
  # -----------------------
  for ($g = 1; $g -le $Generations; $g++) {
    foreach ($r in $Ratios) {
      $rLabel = RatioLabel $r
      $label  = ("gen{0:D2}_{1}" -f $g, $rLabel)

      $realShard   = (".\data\real_train_gen{0:D2}.jsonl" -f $g)
      $promptsShard = (".\data\train_prompts_gen{0:D2}.jsonl" -f $g)

      if (-not (Test-Path $realShard)) { throw "Missing real shard: $realShard" }
      if (-not (Test-Path $promptsShard)) { throw "Missing prompts shard: $promptsShard" }

      # previous adapter (gen00 for g=1, else prev gen same ratio)
      if ($g -eq 1) {
        $prevLabel   = "gen00"
        $prevAdapter = ".\runs\gen00_adapter"
      } else {
        $prevLabel   = ("gen{0:D2}_{1}" -f ($g-1), $rLabel)
        $prevAdapter = (".\runs\{0}_adapter" -f $prevLabel)
      }
      if (-not (Test-Path $prevAdapter)) { throw "Missing previous adapter: $prevAdapter" }

      # Only generate as many synth samples as needed for this ratio
      $needSynth = [int]([Math]::Round($TotalN * $r))
      if ($needSynth -lt 0) { $needSynth = 0 }

      $synthFile = (".\data\synth\{0}_to_gen{1:D2}_{2}.jsonl" -f $prevLabel, $g, $rLabel)

      if (-not (Test-Path $synthFile)) {
        Write-Log ("GENERATE synth for {0}: needSynth={1} -> {2}" -f $label, $needSynth, $synthFile)
        & $py -u .\collapse_pipeline.py generate `
          --model $Model `
          --adapter $prevAdapter `
          --prompts_jsonl $promptsShard `
          --out_jsonl $synthFile `
          --max_new_tokens $MaxNewTokens `
          --temperature $Temperature `
          --top_p $TopP `
          --batch_size $GenBatchSize `
          --max_prompts $needSynth `
          --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "generate failed for $label" }
      } else {
        Write-Log ("Synth exists -> skipping: {0}" -f $synthFile)
      }

      $mixedTrain = (".\data\mixed\{0}_train.jsonl" -f $label)
      $mixMeta    = (".\data\mixed\{0}_mix_meta.json" -f $label)

      $nSynth = [int]([Math]::Round($TotalN * $r))
      $nReal  = $TotalN - $nSynth

      if (-not (Test-Path $mixedTrain)) {
        Write-Log ("MIX {0} -> {1}" -f $label, $mixedTrain)
        & $py -u .\collapse_pipeline.py mix `
          --real_train $realShard `
          --synth $synthFile `
          --out_train $mixedTrain `
          --total_n $TotalN `
          --synth_frac $r `
          --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "mix failed for $label" }

        $meta = [ordered]@{
          label        = $label
          generation   = $g
          ratio_label  = $rLabel
          synth_frac   = $r
          total_n      = $TotalN
          n_real       = $nReal
          n_synth      = $nSynth
          real_shard   = (Resolve-Path $realShard).Path
          synth_file   = (Resolve-Path $synthFile).Path
          mixed_train  = (Resolve-Path $mixedTrain).Path
          seed         = $Seed
          created_at   = (NowIso)
        } | ConvertTo-Json -Depth 5

        $meta | Out-File -FilePath $mixMeta -Encoding utf8
        Write-Log ("Wrote mix metadata -> {0}" -f $mixMeta)
      } else {
        Write-Log ("Mixed train exists -> skipping: {0}" -f $mixedTrain)
        if (-not (Test-Path $mixMeta)) {
          $meta = [ordered]@{
            label        = $label
            generation   = $g
            ratio_label  = $rLabel
            synth_frac   = $r
            total_n      = $TotalN
            n_real       = $nReal
            n_synth      = $nSynth
            real_shard   = (Resolve-Path $realShard).Path
            synth_file   = (Resolve-Path $synthFile).Path
            mixed_train  = (Resolve-Path $mixedTrain).Path
            seed         = $Seed
            created_at   = (NowIso)
          } | ConvertTo-Json -Depth 5

          $meta | Out-File -FilePath $mixMeta -Encoding utf8
          Write-Log ("Wrote missing mix metadata -> {0}" -f $mixMeta)
        }
      }

      $adapterOut = (".\runs\{0}_adapter" -f $label)
      if (-not (Test-Path $adapterOut)) {
        Write-Log ("TRAIN {0}" -f $label)
        & $py -u .\collapse_pipeline.py train `
          --model $Model `
          --train_jsonl $mixedTrain `
          --out_adapter $adapterOut `
          --epochs $EpochsNext `
          --batch $Batch `
          --grad_accum $GradAccum `
          --max_length $MaxLength `
          --lr $LR `
          --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "train failed for $label" }
      } else {
        Write-Log ("Adapter exists -> skipping: {0}" -f $adapterOut)
      }

      $metricsOut = (".\metrics\{0}.json" -f $label)
      Write-Log ("EVAL {0} -> {1}" -f $label, $metricsOut)
      & $py -u .\collapse_pipeline.py eval `
        --model $Model `
        --adapter $adapterOut `
        --real_test .\data\real_test.jsonl `
        --anchor_prompts .\data\anchor_prompts.jsonl `
        --out_metrics $metricsOut `
        --max_test_examples 800 `
        --anchor_limit 200 `
        --max_new_tokens $MaxNewTokens `
        --temperature $Temperature `
        --top_p $TopP `
        --max_length $MaxLength `
        --seed $Seed
      if ($LASTEXITCODE -ne 0) { throw "eval failed for $label" }

      Write-Log ("DONE {0}" -f $label)
    }
  }

  Write-Log ("ALL DONE. Log: {0}" -f $LogPath)

} finally {
  Stop-Transcript | Out-Null
}
