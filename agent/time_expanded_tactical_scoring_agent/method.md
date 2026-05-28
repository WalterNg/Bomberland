# Time-Expanded Tactical Scoring Agent

## 1. Mục tiêu

Method này xây một agent luật chiến thuật nhưng không đi theo chuỗi if-else cứng như các baseline đơn giản. Thay vào đó, agent:

- mô hình hóa nguy hiểm theo thời gian,
- đánh giá toàn bộ hành động hợp lệ trong mỗi lượt,
- ưu tiên sống sót trước rồi mới tới item, box và enemy pressure,
- chỉ đặt bomb khi có giá trị chiến thuật thực sự và vẫn có đường thoát.

Mục tiêu thực tế là tạo ra một agent nhẹ, dễ debug, chạy ổn trong giới hạn thời gian của competition, nhưng vẫn mạnh hơn logic ưu tiên tĩnh kiểu baseline.

## 2. Kiến trúc tổng quát

Folder này theo đúng pattern submission dạng `agent.py` + `method.md`, giống như `agent/dqn_agent/` và `agent/qlearning_v1_agent/`.

Các thành phần chính trong `agent.py`:

1. Parser trạng thái từ `obs`.
2. Mô phỏng blast và chain reaction của bomb.
3. Bản đồ nguy hiểm theo thời gian `danger_by_t`.
4. BFS mở rộng theo thời gian trên trạng thái `(x, y, t)`.
5. Bộ sinh hành động hợp lệ.
6. Bộ chấm điểm cho từng hành động.
7. Override khẩn cấp khi đang ở vùng nguy hiểm.
8. Fallback an toàn nếu không có quyết định tốt hơn.

## 3. Parser trạng thái

Agent đọc trực tiếp:

- `map`: grid 13x13,
- `players`: trạng thái của 4 player,
- `bombs`: danh sách bomb hiện tại.

Từ đó agent trích ra:

- vị trí của bản thân,
- số bomb còn lại,
- bonus radius hiện tại,
- vị trí enemy còn sống,
- vị trí bomb trên map,
- các ô passable và các ô bị chặn.

Radius bomb thực tế được tính theo công thức:

```text
actual_radius = 1 + bomb_radius_bonus
```

Để ổn định hơn, agent cache radius của bomb ngay khi bomb xuất hiện lần đầu, thay vì suy ngược lại mỗi lượt.

## 4. Mô hình nguy hiểm theo thời gian

Phần quan trọng nhất của method là không gộp tất cả nguy hiểm vào một tập `danger_soon` thô. Thay vào đó, agent xây:

```text
danger_by_t[t] = tập các ô sẽ nổ tại thời điểm t
```

Trong code, horizon mặc định là 8 bước. Điều này đủ để bao phủ timer bomb chuẩn và các quyết định thoát hiểm ngắn hạn.

### 4.1 Blast tiles

Blast của bomb là cross shape:

- gồm ô bomb,
- lan theo 4 hướng,
- dừng ở wall,
- dừng sau box,
- không dừng ở player.

### 4.2 Chain reaction

Nếu blast của bomb A chạm vào bomb B trước khi B tự nổ, B sẽ bị kéo nổ sớm hơn. Agent xử lý bằng vòng lặp lan truyền thời gian nổ cho tới khi ổn định.

Điểm này quan trọng vì thời gian nổ thực tế có thể khác timer gốc nếu có chain reaction.

## 5. Time-expanded BFS

Thay vì BFS chỉ trên `(x, y)`, agent search trên:

```text
(x, y, t)
```

vì một ô có thể an toàn ở thời điểm này nhưng nguy hiểm ở vài bước sau.

Search này được dùng cho:

- kiểm tra có escape path hay không,
- đếm safe area,
- ước lượng độ sâu an toàn của vị trí,
- tìm target item/box/enemy theo đường đi an toàn.

### 5.1 Điều kiện chuyển trạng thái

Một bước đi hợp lệ khi:

- ô đích nằm trong board,
- ô đích passable,
- ô đích không bị bomb hoặc enemy chiếm,
- ô đích không nằm trong `danger_by_t` tại thời điểm đến.

Agent cho phép action `STOP` trong BFS để mô phỏng việc chờ đợi an toàn.

### 5.2 Safe area

Không phải ô nào reachable cũng được tính là “safe tile” dùng để đánh giá escape. Agent chỉ tính những ô có một khoảng an toàn tối thiểu trong vài bước tiếp theo. Điều này giúp tránh các ngõ cụt hoặc ô sắp nổ ngay sau khi đến.

## 6. Chấm điểm hành động

Agent chấm điểm tất cả action hợp lệ:

- `STOP`
- `LEFT`
- `RIGHT`
- `UP`
- `DOWN`
- `PLACE_BOMB`

### 6.1 Điểm của move

Điểm move được ghép từ:

- khả năng sống sót,
- số ô an toàn có thể tiếp cận,
- khoảng cách tới ô an toàn tốt nhất,
- độ mở xung quanh vị trí mới,
- item target score,
- box pressure score,
- enemy pressure score,
- penalty nếu ô đó sắp nguy hiểm.

Ý tưởng là một move tốt không chỉ “chưa chết ngay”, mà còn phải dẫn tới vùng có nhiều lựa chọn tiếp theo.

### 6.2 Điểm của bomb

Bomb chỉ được giữ lại nếu:

- còn bomb để đặt,
- không có bomb ngay trên ô hiện tại,
- sau khi đặt bomb vẫn có đường thoát,
- tổng giá trị chiến thuật dương.

Giá trị bomb gồm:

- số box phá được,
- giá trị trap enemy,
- chain reaction value,
- trừ chi phí escape,
- trừ self risk,
- trừ waste penalty nếu bomb không tạo ra giá trị.

## 7. Ưu tiên mục tiêu

Agent không dùng target cứng. Thay vào đó, target được đổi thành score theo ngữ cảnh.

### 7.1 Item

Item capacity thường có giá trị cao hơn khi còn ít bomb. Item radius có giá trị giảm dần khi radius đã đủ lớn.

### 7.2 Box spots

Một ô được xem là box spot nếu đứng ở đó có thể đặt bomb phá được ít nhất một box. Spot nào phá được nhiều box hơn sẽ được ưu tiên hơn.

### 7.3 Enemy pressure

Agent không lao thẳng vào enemy bằng mọi giá. Nó chỉ cộng điểm pressure khi vị trí mới vẫn an toàn và có đường rút.

## 8. Override khẩn cấp

Nếu vị trí hiện tại đang rơi vào danger ngay hoặc rất gần danger, agent bỏ qua phần scoring bình thường và chỉ chọn hành động thoát hiểm tốt nhất.

Trong chế độ này:

- bomb không được ưu tiên,
- stop bị hạn chế,
- tiêu chí số một là tìm ô có safe margin tốt nhất.

## 9. Fallback policy

Nếu tất cả chấm điểm đều xấu, agent sẽ:

1. chọn move an toàn nhất còn lại,
2. nếu không có move nào an toàn thì STOP,
3. không đặt bomb làm fallback mù.

## 10. Hyperparameters chính

Các tham số chính hiện được tách ra file `config.yaml` cùng thư mục. File này chứa 3 profile:

- `safe`
- `balanced`
- `aggressive`

 Chiến thuật động được tách sang file `strategy.py` với class `Strategy`. Class này theo dõi số đối thủ còn lại, số box còn lại và item gần nhất để chuyển qua các nhịp `collector -> pressure -> siege -> solo_spam` theo từng turn. Khi nhiều đối thủ và box còn nhiều, nó nghiêng về phá box và nhặt item; giai đoạn `collector` dùng profile riêng để đầu game chịu đẩy bomb và ưu tiên hộp/item rõ hơn; khi số đối thủ hoặc box giảm, nó nâng mức aggressive dần lên; khi còn tối đa 1 đối thủ, nó bật chế độ spam bomb theo burst tối đa 3 quả và tạm dừng spam nếu có item gần.

Agent khởi đầu ở profile `balanced` để vừa giữ độ an toàn vừa chủ động đi nhặt item và tìm cơ hội tạo lợi thế ở đầu trận. Khi box giảm dần hoặc số đối thủ ít đi, nó sẽ đẩy mức aggressive lên. Khi trên sân chỉ còn 2 agent sống, agent tự chuyển sang chế độ `aggressive` mạnh tay hơn để spam bomb và truy đuổi đối thủ. Nếu muốn ép hành vi mặc định khác, chỉ cần sửa hằng số `PROFILE_NAME` ở đầu `agent.py`:

```text
PROFILE_NAME = "balanced"
```

Ngoài các weight chính, config còn có `BOMB_DECISION_MARGIN` để cho phép chọn bomb ngay cả khi bomb score chỉ kém move an toàn một chút.
`BOMB_TACTICAL_BONUS` là phần cộng thêm cho bomb khi nó có giá trị chiến thuật thật sự, giúp bomb không bị thua quá dễ trước các move an toàn.

Các tham số chính trong config:

- `HORIZON = 8`
- `SAFE_MARGIN = 2`
- `DEATH_PENALTY = -1_000_000`
- `NO_ESCAPE_PENALTY = -100_000`
- `DANGER_NEXT_STEP_PENALTY = -50_000`
- `DANGER_SOON_PENALTY = -5_000`
- `STOP_PENALTY = -10`
- `BOX_VALUE = 60`
- `MULTI_BOX_BONUS = 30`
- `ENEMY_HIT_BONUS = 300`
- `ITEM_SCORE_MULTIPLIER` phóng đại ưu tiên item trong move scoring.
- `ITEM_PROGRESS_BONUS` và `ITEM_STEP_BONUS` giúp agent chủ động tiến về item thay vì đứng im.
- `STOP_AVOID_MARGIN` ép agent đi thay vì đứng yên nếu STOP chỉ nhỉnh hơn move khác một chút.

Các giá trị này được chọn theo hướng an toàn trước, sau đó mới mở rộng khả năng farm box và ép enemy.

## 11. Điểm mạnh của method

Method này tốt hơn baseline rule-chain ở các điểm sau:

- phân biệt rõ thời điểm nguy hiểm,
- tránh over-avoid các ô thực ra còn an toàn,
- đánh giá bomb theo giá trị kỳ vọng thay vì chỉ check hit đơn giản,
- giữ được tính deterministic và dễ debug,
- vẫn nhẹ đủ để phù hợp môi trường inference giới hạn thời gian.

## 12. Gợi ý tinh chỉnh

Nếu muốn nâng tiếp, ưu tiên tinh chỉnh theo thứ tự:

1. weight của item vs box vs enemy,
2. safe margin của BFS,
3. ngưỡng `waste_penalty` cho bomb,
4. late-game multiplier,
5. logic enemy trap approximation.

Mục tiêu khi tuning là tăng score mà không làm self-kill hoặc timeout tăng lên.

Trong giai đoạn solo cuối, agent ưu tiên bomb chủ động hơn và sau khi đặt bomb sẽ đẩy trọng số nghiêng về việc chạy xa khỏi vùng nổ trong vài bước tiếp theo.
Nếu bomb vẫn có đường thoát hợp lệ, solo aggressive sẽ ưu tiên ném bomb ngay thay vì tiếp tục so giữa bomb và move như ở đầu trận.
Agent cũng có nhịp bomb theo chu kỳ: sau khi vừa đặt bomb, nó phải rời xa vùng bomb gần nhất một đoạn tối thiểu trước khi được phép đặt bomb tiếp.
Nếu đang tiến gần item trong solo cuối, bomb spam sẽ tạm dừng cho tới khi item đó được nhặt xong hoặc không còn item gần nữa.
