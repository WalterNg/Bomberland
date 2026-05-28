# Q-Learning V1 cho Bomberland

## 1) Mục tiêu

Mục tiêu của `qlearning_v1_agent` là tạo một baseline Q-learning có thể train local, đúng giao diện submission của cuộc thi, và có state space rời rạc để cập nhật bằng Q-table.

## 2) Kiến trúc code

Folder:

- `agent/qlearning_v1_agent/agent.py`: runtime `Agent` + Q-learning core + train entrypoint.
- `agent/qlearning_v1_agent/method.md`: tài liệu phương pháp và kết quả.

Trong runtime submission, class `Agent`:

- đọc `q_table.json` nếu tồn tại.
- tạo state rời rạc từ `obs`.
- chọn hành động bằng epsilon-greedy (inference đặt `epsilon=0.0`).

## 3) Thiết kế state rời rạc

State được mã hóa thành tuple 8 biến rời rạc:

1. `safe_tile_flag` (0/1): vị trí hiện tại có an toàn trước nổ hay không.
2. `bomb_available_flag` (0/1): còn bomb để đặt hay không.
3. `bomb_threat_bucket` (0..3): mức độ gấp theo timer nhỏ nhất của các bomb có blast chạm vị trí hiện tại.
4. `enemy_dir_bucket` (0..4): hướng tới enemy gần nhất.
5. `enemy_dist_bucket` (0..4): khoảng cách bucket tới enemy gần nhất.
6. `box_dir_bucket` (0..4): hướng tới box gần nhất.
7. `box_dist_bucket` (0..4): khoảng cách bucket tới box gần nhất.
8. `local_adjacency_code`: mã hóa 4 ô xung quanh (up/down/left/right) thành mã số rời rạc.

State key được serialize thành chuỗi để lưu trong Q-table dictionary.

## 4) Bomb threat và radius cache

`obs["bombs"]` không chứa radius trực tiếp, nên agent dùng cache theo từng bomb:

- Khi bomb mới xuất hiện (có trong `curr_obs` nhưng không có trong frame trước), agent lưu radius snapshot:
  - `radius = 1 + players[owner_id][4]` tại thời điểm bomb vừa xuất hiện.
- Radius này được giữ cho đến khi bomb biến mất.
- Khi tính threat, agent dùng cache radius để suy ra blast tiles và tìm `t_min`.

Cách này giúp threat bucket rời rạc ổn định hơn so với việc suy theo radius hiện tại của owner.

## 5) Learning update

Q-learning update:

`Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))`

Reward shaping v1 tập trung vào:

- sống/chết
- enemy chết
- thắng trận
- item collection
- movement/time penalty nhẹ

## 6) Lệnh train local

Lệnh train đã dùng:

```powershell
python -m agent.qlearning_v1_agent.agent --train --episodes 600 --enemy_type simple --seed 42 --alpha 0.1 --gamma 0.95 --epsilon_start 1.0 --epsilon_min 0.05 --epsilon_decay 0.999
```

File output:

- `agent/qlearning_v1_agent/q_table.json`

## 7) Lệnh evaluate 100 matches

Lệnh đánh giá nên dùng (script mặc định của repo):

```powershell
python -m scripts.participant.estimate_rankings --agent_path agent/qlearning_v1_agent/ --num_matches 100
```

Pool đối thủ của script này:

- `TacticalRuleAgent`
- `SmarterRuleAgent`
- `GeniusRuleAgent`

## 8) Kết quả benchmark

Đánh giá bằng nhiều tổ hợp hyper-params, mỗi cấu hình:

- train local với `--episodes 600 --enemy_type simple --seed 42`
- evaluate bằng `estimate_rankings` với `--num_matches 100`

| Config | alpha | gamma | epsilon_decay | epsilon_min | episodes | Win Rate | Draw Rate | Avg Rank | Score | mu | sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_base | 0.10 | 0.95 | 0.9990 | 0.05 | 600 | 1.0% | 0.0% | 2.66 | 45.22 | 54.92 | 3.23 |
| B_high_alpha | 0.20 | 0.95 | 0.9990 | 0.05 | 600 | 2.0% | 0.0% | 2.44 | 56.28 | 65.61 | 3.11 |
| C_high_gamma_slow_decay | 0.05 | 0.99 | 0.9995 | 0.02 | 600 | 3.0% | 0.0% | 2.65 | 44.05 | 54.56 | 3.50 |
| D_high_gamma | 0.10 | 0.99 | 0.9990 | 0.05 | 600 | 0.0% | 0.0% | 2.68 | 38.70 | 49.95 | 3.75 |
| E_fast_decay_low_gamma | 0.10 | 0.90 | 0.9980 | 0.05 | 600 | 0.0% | 0.0% | 2.70 | 37.02 | 48.17 | 3.72 |
| F_alpha025_gamma095 | 0.25 | 0.95 | 0.9990 | 0.05 | 600 | 1.0% | 1.0% | 2.61 | 46.13 | 55.77 | 3.21 |
| G_alpha015_gamma095 | 0.15 | 0.95 | 0.9990 | 0.05 | 600 | 1.0% | 0.0% | 2.68 | 38.33 | 49.21 | 3.63 |
| H_alpha020_gamma097 | 0.20 | 0.97 | 0.9990 | 0.05 | 600 | 1.0% | 0.0% | 2.67 | 39.28 | 50.14 | 3.62 |
| I_alpha020_gamma093 | 0.20 | 0.93 | 0.9990 | 0.05 | 600 | 1.0% | 0.0% | 2.67 | 39.56 | 50.39 | 3.61 |
| J_alpha020_more_episodes | 0.20 | 0.95 | 0.9990 | 0.05 | 1000 | 1.0% | 0.0% | 2.75 | 35.88 | 46.48 | 3.53 |

Best theo `Score`: `B_high_alpha` (Score `56.28`, Avg Rank `2.44`).

Nhận xét nhanh:

- Bản `qlearning_v1_agent` hiện tại đã chạy ổn định end-to-end và có thể submit được.
- Chất lượng chơi chưa cao khi benchmark với đối thủ mạnh.
- Tăng `alpha` từ `0.10` lên `0.20` cho kết quả tốt hơn rõ rệt trong sweep này.
- Với dữ liệu hiện tại, tăng thêm episodes lên `1000` chưa giúp tốt hơn.
- Bước tiếp theo nên làm là tăng chất lượng reward shaping và state feature, sau đó train với curriculum đối thủ khó dần.
