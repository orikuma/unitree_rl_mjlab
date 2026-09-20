# G1 の5段階段上り

`Unitree-G1-Stairs` は、`main` の G1 rough task を基に、平地と階段を同じ
PPO policy で学習する環境です。最終目標は蹴上げ17 cm・踏面17 cmの5段
（総上昇85 cm）を下の平地から登り、上の踊り場へ移ることです。
G1 のモデル、29関節の位置指令、PD制御、action scale、PPOと既存のexport処理を再利用します。
Runnerには環境ごとのカリキュラム状態をcheckpointへ保存・復元する小さな拡張を加えています。
学習済みpolicyは含みません。実際の達成率は、以下の独立評価で確認してください。

## 学習

リポジトリのルートで、既存の mjlab 環境を有効にして実行します。

```bash
PYTHONPATH=. python scripts/train.py Unitree-G1-Stairs \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.run-name stairs_seed42 \
  --agent.seed 42
```

GPUメモリに応じて並列環境数を調整できます。ログとcheckpointは
`logs/rsl_rl/g1_stairs_perceptive/<日時>_stairs_seed42/` に保存されます。
`train_stairs_pipeline.sh` や段階ごとにpolicyを作り直す処理は使いません。
平地歩行・停止の合格を待つことなく、最初から低い5段階段を経験します。

## 地形とカリキュラム

階段幅は1.2 m、初段は環境原点の0.6 m前方にあります。
開始位置の前後方向を±0.2 m変化させるので、初段までの距離は0.4～0.8 mです。
左右位置・向きにも小さなばらつきを与え、前進指令は0.25～0.4 m/sとします。
第5段の後には長さ1.2 m以上の踊り場があります。

| Level | 段数 | 蹴上げ cm | 踏面 cm |
| --- | ---: | ---: | ---: |
| 0 | 0（平地） | 0 | — |
| 1 | 5 | 2 | 30 |
| 2 | 5 | 4 | 30 |
| 3 | 5 | 6 | 30 |
| 4 | 5 | 8 | 30 |
| 5 | 5 | 10 | 30 |
| 6 | 5 | 12 | 30 |
| 7 | 5 | 14 | 30 |
| 8 | 5 | 17 | 30 |
| 9 | 5 | 17 | 25 |
| 10 | 5 | 17 | 21 |
| 11 | 5 | 17 | 19 |
| 12 | 5 | 17 | 17 |

各並列環境が個別に現在の課題levelを持ち、そのlevelで成功すると1段階昇格します。
同じ課題で3回続けて成功できなければ1段階戻ります。時間切れも不成功に含めます。
現在の課題がlevel 1なら約30%のepisodeを平地、70%をlevel 1から採取します。
昇格後は約20%を平地、20%を現在より易しい階段、60%を現在の課題とします。
比率はreset時の確率で、すべて現在のpolicyによる新しいrolloutです。
地形の変更時にpolicy・optimizer・観測正規化を初期化しません。

寸法は `src/tasks/stairs/terrains.py` に明示した値です。連続的な難度補間に
最終寸法を任せません。足裏形状やロボットの衝突形状は変更していません。
17 cmの踏面では踵などが張り出す部分支持を許し、足裏全体の踏面内への包含は要求しません。

## 観測・報酬・終了条件

actorは既存のIMU・関節状態・直前actionに加えて、5 cm間隔の高さマップを観測します。
固有感覚は5フレームの履歴、高さマップは現在の値を使います。
raycastは胴体のIMU site（`imu_in_torso`）から地形のgeometry group 0だけへ向けます。
criticには既存の追加情報と、成功判定に必要な支持・保持時間を渡します。

通常学習の固有感覚ノイズはrough taskの1/4、高さノイズは±5 mmです。
初期学習では摩擦係数を0.9に固定し、外力、重心変動、encoder biasを除きます。
`Unitree-G1-Stairs-Robust` では固有感覚ノイズを元の大きさ、高さノイズを±2 cmにし、
摩擦0.6～1.2、重心±2 cm、encoder biasの変動を有効にします。外力は加えません。

速度報酬はyawだけで回転した水平速度を使い、上下方向の速度を罰しません。
静止時の速度追従値を差し引き、前後方向の符号付き進捗を加えます。
世界Z座標の足先高さ、固定周期の接地パターン、下肢の既定姿勢への拘束、停止報酬を外し、
上体の軽い正則化と既存の滑り・過大な関節動作などの費用を残します。
完了は一度だけ+5、失敗は−5です。標準RewardManagerの `dt` 倍を相殺するので、
この2項は時間刻みで小さくなりません。

足ごとに強い接触を最大4点取得し、接触法線、力、位置で踏面上の支持を判定します。
蹴上げへの衝突は支持として扱いません。両足が上側に移動し、踊り場に到達してから、
交互支持を許して1秒間安定を維持すると成功です。精密停止は要求しません。
大きな傾き、階段通路からの逸脱、足以外の継続的な地形接触を失敗とし、20秒で時間切れにします。

## 再生と独立評価

再生はデフォルトで正確な17×17 cm・5段を使用します。

```bash
PYTHONPATH=. python scripts/play.py Unitree-G1-Stairs \
  --checkpoint-file logs/rsl_rl/g1_stairs_perceptive/RUN/model_N.pt
```

`RUN` と `model_N.pt` は実際のrun名とcheckpoint名に置き換えます。
基本評価は平均行動・観測ノイズなしで、平地、低い階段、高い広い階段、最終寸法を比較します。

```bash
PYTHONPATH=. python scripts/evaluate_stairs.py \
  --checkpoint-file logs/rsl_rl/g1_stairs_perceptive/RUN/model_N.pt \
  --levels "(0, 1, 8, 12)" --seeds "(101, 202, 303)" \
  --num-envs 32 --episodes 100 \
  --output reports/stairs_basic.json
```

`--episodes` は各level・各seedのepisode数です。この例では計1,200 episodeです。
環境ごとに評価数を割り当てるので、早く失敗する環境だけが集計を占めることを防ぎます。
最終寸法の頑健性は別に評価します。

```bash
PYTHONPATH=. python scripts/evaluate_stairs.py \
  --checkpoint-file logs/rsl_rl/g1_stairs_perceptive/RUN/model_N.pt \
  --levels "(12,)" --seeds "(101, 202, 303)" --num-envs 32 --episodes 100 \
  --robust True --observation-noise 1.0 \
  --output reports/stairs_robust.json
```

`--observation-noise` は選択した設定のノイズ倍率で、通常設定のノイズだけを調べるなら
`--robust True` を外して `--observation-noise 1.0` を指定します。
JSONには成功率と95% Wilson区間、各段への支持到達数、失敗・時間切れ、前進距離、
速度追従のRMSE、蹴上げ衝突の時間割合、足以外の接触、seed、設定、checkpointのSHA-256を保存します。
学習ログにも `Stairs/level_<N>/success` などを記録します。
平均報酬や途中の段への到達だけで、最終5段の成功と判断しないでください。

## 頑健性の追加学習と再開

最終寸法の登段を学習できたcheckpointから、同じ観測・行動構成のまま
`Unitree-G1-Stairs-Robust` で頑健性を追加学習できます。
平地・易しい階段・各環境の現在の課題を混合し続け、既に習得した歩行も維持します。

```bash
PYTHONPATH=. python scripts/train.py Unitree-G1-Stairs-Robust \
  --env.scene.num-envs 1024 --agent.logger tensorboard \
  --agent.resume True --agent.load-run RUN --agent.load-checkpoint model_N.pt \
  --agent.run-name stairs_robust_seed42 --agent.seed 42
```

階段用Runnerはpolicy・optimizer・観測正規化とともに、環境ごとの現在の課題levelと
連続不成功回数を保存・復元します。通常のノイズ設定で学習を再開する場合は、上のtask名を
`Unitree-G1-Stairs` に変更して再開します。シミュレータのepisode途中の状態までを
完全に再現する再開ではありません。
Robust taskも保存された各環境の課題levelと連続不成功回数を引き継ぎ、同じカリキュラムを使います。
再生は通常taskと同じくlevel 12が標準です。追加学習後にも平地と最終寸法の両方を評価してください。

## 設計の参考

- [Siekmann et al., Blind Bipedal Stair Traversal via Sim-to-Real Reinforcement Learning](https://arxiv.org/abs/2105.08328): 二足の階段訓練と時間的情報の利用。
- [Rudin et al., Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html): 並列PPOと地形カリキュラム。
- [Miki et al., Learning robust perceptive locomotion for quadrupedal robots in the wild](https://doi.org/10.1126/scirobotics.abk2822): 地形観測と固有感覚の併用。

これらは設計原則の参考であり、このG1・階段寸法の成功を保証するものではありません。
実機への展開には、同じ高さマップと観測履歴を供給する知覚・制御処理が別途必要です。
