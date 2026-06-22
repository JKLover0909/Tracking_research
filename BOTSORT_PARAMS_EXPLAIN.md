# Giải Thích Các Tham Số Trong `bytetrack.yaml`

Tài liệu này giải thích các tham số tracker đang được dùng trong file [bytetrack.yaml](/mnt/nvme/opt/Tracking_research/bytetrack.yaml), theo hướng dễ đưa vào báo cáo kỹ thuật.

## 1. Tổng quan

Cấu hình hiện tại đang dùng:

```yaml
tracker_type: botsort
track_high_thresh: 0.35
track_low_thresh: 0.2
new_track_thresh: 0.40
track_buffer: 120
match_thresh: 0.82
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.6
appearance_thresh: 0.35
with_reid: True
model: auto
```

`BoT-SORT` là một thuật toán multi-object tracking được thiết kế để giảm `ID switch` và giữ định danh ổn định hơn trong các bài toán theo dõi người. Nó kết hợp:

- thông tin vị trí và bounding box
- độ giống nhau về appearance (ReID)
- bù chuyển động camera (GMC)
- cơ chế ghép track qua nhiều ngưỡng confidence

Trong bài toán theo dõi người mặc đồng phục, `BoT-SORT` hữu ích hơn tracker đơn thuần theo vị trí vì nhiều đối tượng có hình ảnh gần giống nhau.

## 2. Luồng hoạt động tổng quát của BoT-SORT

Mỗi frame, tracker xử lý theo logic gần đúng như sau:

1. Nhận danh sách detection từ model detector.
2. Tách detection thành nhiều nhóm theo confidence.
3. Cố gắng ghép detection mới với các track đang tồn tại.
4. Nếu bật `ReID`, tracker kiểm tra thêm độ giống nhau về appearance.
5. Nếu camera có chuyển động, áp dụng `GMC` để bù lại sai lệch do camera gây ra.
6. Track nào không thấy trong vài frame vẫn được giữ tạm thời trong bộ đệm.
7. Detection nào không ghép được với track cũ, nếu đạt ngưỡng tạo track mới thì tạo ID mới.

Nói ngắn gọn:

`Detection -> lọc theo confidence -> matching với track cũ -> kiểm tra appearance/vị trí -> cập nhật track hoặc tạo track mới`

## 3. Giải thích từng tham số

### `tracker_type: botsort`

Tham số này chỉ ra tracker được sử dụng là `BoT-SORT`.

Ý nghĩa:

- Sử dụng chiến lược tracking của BoT-SORT thay vì ByteTrack thường.
- Hỗ trợ kết hợp thêm `ReID` và `GMC`.
- Phù hợp hơn khi cần giảm nhầm ID trong các cảnh có nhiều người giống nhau.

Tác động:

- Tăng độ ổn định ID.
- Thường tốn thêm tài nguyên tính toán so với tracker đơn giản hơn.

### `track_high_thresh: 0.35`

Ngưỡng confidence cao để ưu tiên cho vòng matching chính.

Ý nghĩa:

- Các detection có confidence lớn hơn hoặc bằng `0.35` được xem là detection tin cậy cao.
- Tracker ưu tiên dùng nhóm detection này để ghép với các track đang tồn tại ở bước đầu.

Tác động:

- Ngưỡng cao hơn: giảm false positive, nhưng có thể bỏ sót đối tượng mờ hoặc khó thấy.
- Ngưỡng thấp hơn: bắt được nhiều đối tượng hơn, nhưng dễ đưa nhầm detection kém chất lượng vào tracker.

Trong cấu hình này, `0.35` là mức khá mềm, ưu tiên giữ track trong các tình huống detect không quá sắc nét.

### `track_low_thresh: 0.2`

Ngưỡng confidence thấp để giữ lại các detection yếu cho vòng ghép phụ.

Ý nghĩa:

- Detection trong khoảng từ `0.2` đến dưới `0.35` không đủ mạnh cho matching chính.
- Tuy vậy, tracker vẫn có thể dùng chúng ở các bước ghép bổ sung để tránh mất người tạm thời.

Tác động:

- Giúp nối tiếp track khi đối tượng bị che một phần, mờ, hoặc xa camera.
- Nếu đặt quá thấp, dễ đưa nhiều detection nhiễu vào tracker.

Thông thường:

- `track_high_thresh` dùng cho matching ưu tiên.
- `track_low_thresh` là lưới an toàn để hạn chế mất track.

### `new_track_thresh: 0.40`

Ngưỡng để khởi tạo một track mới.

Ý nghĩa:

- Detection chỉ được tạo thành ID mới nếu confidence đạt từ `0.40` trở lên.
- Dùng để tránh việc các detection quá yếu tạo ra rất nhiều track ảo.

Tác động:

- Ngưỡng cao hơn: giảm track giả, nhưng dễ bỏ sót đối tượng mới vào khung hình.
- Ngưỡng thấp hơn: phát hiện đối tượng mới nhanh hơn, nhưng dễ sinh nhiều ID rác.

Lưu ý:

- `new_track_thresh` thường nên bằng hoặc cao hơn `track_high_thresh`.
- Ở đây `0.40 > 0.35`, nghĩa là match track cũ thì mềm hơn, nhưng tạo track mới thì chặt hơn.

### `track_buffer: 120`

Số frame tối đa track có thể "mất dấu tạm thời" trước khi bị xóa.

Ý nghĩa:

- Nếu đối tượng tạm thời biến mất, track không bị xóa ngay.
- Tracker giữ nó trong `120` frame để chờ cơ hội nối lại với detection ở các frame sau.

Tác động:

- Giá trị lớn giúp giảm `ID switch` khi đối tượng bị che khuất, ra vào khung hình ngắn hạn.
- Giá trị quá lớn có thể giữ lại nhiều track cũ, tăng nguy cơ ghép nhầm nếu cảnh đông.

Với video `30 FPS`, `120` frame tương đương khoảng `4 giây`.

### `match_thresh: 0.82`

Ngưỡng match giữa detection và track cũ.

Ý nghĩa:

- Thể hiện mức độ để tracker chấp nhận một cặp ghép.
- Giá trị này thường được dùng trên ma trận độ giống nhau hoặc khoảng cách sau khi đã kết hợp các thông tin tracking.

Tác động:

- Giá trị cao hơn: tracker khó tính hơn khi ghép, giảm ghép nhầm.
- Giá trị thấp hơn: tracker dễ ghép hơn, giúp nối track nhanh nhưng tăng nguy cơ sai ID.

Trong thực tế:

- `0.82` là mức tương đối chặt.
- Phù hợp khi ưu tiên độ chính xác ID hơn là bắt bằng mọi detection.

### `fuse_score: True`

Cho phép kết hợp confidence detection vào quá trình matching.

Ý nghĩa:

- Tracker không chỉ nhìn vị trí/IoU, mà còn cân nhắc độ tin cậy của detection.
- Detection có confidence cao thường được ưu tiên hơn trong matching.

Tác động:

- Giảm khả năng tracker bám vào detection yếu, nhiễu.
- Làm matching ổn định hơn khi detector cho chất lượng đầu ra không đồng đều.

Nếu tắt:

- Tracking sẽ dựa nhiều hơn vào thông tin hình học và ReID.

### `gmc_method: sparseOptFlow`

Sử dụng `Global Motion Compensation` bằng sparse optical flow.

Ý nghĩa:

- Khi camera tự rung, lia, hoặc quay, bounding box của mỗi người đều có vẻ như đang di chuyển mạnh.
- `GMC` cố gắng ước lượng chuyển động chung của toàn cảnh rồi bù trước khi matching.

Tác động:

- Giảm `ID switch` do camera rung hoặc pan/tilt.
- Hữu ích cho camera giám sát không cố định tuyệt đối.

`sparseOptFlow` hoạt động bằng cách:

- chọn các điểm đặc trưng trong ảnh
- theo dõi sự di chuyển của chúng qua hai frame
- ước lượng chuyển động tổng quát của camera
- dùng thông tin này để hiểu "chuyển động nào là do camera, chuyển động nào là do đối tượng"

Nếu camera cố định rất chắc, tác dụng của tham số này sẽ nhỏ hơn.

### `proximity_thresh: 0.6`

Ngưỡng gần nhau về không gian để cho phép so khớp ReID.

Ý nghĩa:

- Dù appearance có giống, tracker vẫn yêu cầu detection mới phải đủ gần track cũ về vị trí.
- Điều này tránh việc ghép hai người ở hai vị trí xa nhau chỉ vì nhìn giống nhau.

Tác động:

- Ngưỡng chặt hơn: giảm ghép sai xa vị trí, nhưng dễ bỏ lỡ trường hợp đối tượng di chuyển nhanh.
- Ngưỡng lỏng hơn: tăng khả năng nối lại track, nhưng dễ nhầm giữa nhiều người giống nhau.

Nó đóng vai trò như một "cửa lọc không gian" trước khi tin vào appearance.

### `appearance_thresh: 0.35`

Ngưỡng giống nhau về appearance để tracker chấp nhận matching khi bật ReID.

Ý nghĩa:

- Detection mới phải đủ giống track cũ về đặc trưng hình ảnh.
- Đặc trưng này thường là embedding vector do model ReID tạo ra.

Tác động:

- Ngưỡng cao hơn: cần giống nhiều mới ghép, an toàn hơn.
- Ngưỡng thấp hơn: dễ nối track hơn, nhưng tăng nguy cơ ghép nhầm người mặc giống nhau.

Với bài toán đồng phục:

- Đây là tham số cần cân nhắc kỹ.
- Nếu quá thấp, nhiều người mặc giống nhau sẽ bị ghép nhầm.
- Nếu quá cao, người cùng một ID nhưng thay đổi tư thế, ánh sáng, góc nhìn có thể bị tách thành ID mới.

### `with_reid: True`

Bật tính năng `ReID`.

Ý nghĩa:

- Tracker sẽ không chỉ dựa vào vị trí mà còn so sánh appearance embedding.
- Rất quan trọng khi đối tượng bị che khuất, cắt ngang, hoặc tracker tạm thời mất box.

Tác động:

- Giảm `ID switch` trong nhiều tình huống khó.
- Tăng chi phí tính toán.

Nếu tắt:

- Tracker gần như chỉ dựa vào thông tin hình học và chuyển động.
- Trong bài toán nhiều người mặc giống nhau, chất lượng ID thường giảm.

### `model: auto`

Lựa chọn model ReID tự động.

Ý nghĩa:

- Hệ thống tự quyết định cách lấy appearance feature phù hợp với pipeline hiện tại.
- Trong một số trường hợp, Ultralytics có thể dùng embedding có sẵn hoặc chọn model ReID phù hợp mà không cần chỉ định thủ công.

Tác động:

- Dễ cấu hình hơn.
- Nhanh để thử nghiệm.

Hạn chế:

- Không kiểm soát được kiến trúc ReID cụ thể bằng cách chỉ định model riêng.
- Nếu cần tối ưu mạnh cho bài toán đồng phục, thường sau này sẽ cần model ReID huấn luyện riêng.

## 4. Mối quan hệ giữa các tham số

Những tham số này không hoạt động độc lập. Chuyển động của tracker là kết quả của nhiều ngưỡng cùng lúc:

- `track_high_thresh` quyết định detection nào đủ tốt để ưu tiên match.
- `track_low_thresh` giữ lại detection yếu để cứu track.
- `new_track_thresh` quyết định khi nào mới tạo ID mới.
- `track_buffer` quyết định track có được chờ đợi để nối lại hay không.
- `match_thresh` quyết định match có đủ tin cậy hay không.
- `proximity_thresh` và `appearance_thresh` ràng buộc đồng thời vị trí và appearance.

Có thể hiểu đơn giản:

- Ngưỡng detect ảnh hưởng đến việc "có nhìn thấy đối tượng không"
- Ngưỡng match ảnh hưởng đến việc "có coi đó là cùng một người không"
- `track_buffer` ảnh hưởng đến việc "có sẵn sàng chờ đối tượng quay lại không"

## 5. Đánh giá bộ tham số hiện tại

Bộ tham số này có xu hướng:

- ưu tiên giữ ID lâu
- chấp nhận detection confidence tương đối thấp để tránh mất track
- bật ReID để giảm `ID switch`
- bật GMC để hỗ trợ khi camera chuyển động

Phù hợp với bài toán:

- tracking người
- nhiều đối tượng có ngoại hình giống nhau
- cần giữ ID ổn định hơn là chỉ cần FPS cao

Rủi ro có thể gặp:

- `track_buffer=120` khá dài, nếu cảnh rất đông có thể giữ track cũ quá lâu
- `appearance_thresh=0.35` cần kiểm thử kỹ nếu đồng phục quá giống nhau
- `with_reid=True` sẽ tốn thêm tài nguyên, đặc biệt trên thiết bị edge

## 6. Gợi ý cách trình bày trong báo cáo

Bạn có thể mô tả ngắn gọn như sau:

> Hệ thống tracking sử dụng BoT-SORT với ReID và Global Motion Compensation. Các ngưỡng confidence được chia thành ngưỡng match chính, ngưỡng cứu track, và ngưỡng tạo track mới. Các track bị mất tạm thời vẫn được duy trì trong 120 frame để giảm ID switch. Quá trình matching không chỉ dựa trên vị trí mà còn kết hợp độ tương đồng appearance, đồng thời áp dụng ràng buộc proximity để tránh ghép nhầm các đối tượng ở xa nhau.

## 7. Kết luận

Bộ tham số hiện tại được chỉnh theo hướng ưu tiên độ ổn định ID trong bài toán tracking người. Điểm mạnh chính nằm ở 3 thành phần:

- `BoT-SORT` để tracking ổn định hơn
- `ReID` để phân biệt đối tượng khi có che khuất hoặc đổi ID tạm thời
- `GMC` để giảm sai số khi camera có chuyển động

Nếu mục tiêu sau này là tối ưu thêm, 3 tham số nên thử nghiệm đầu tiên là:

- `track_buffer`
- `appearance_thresh`
- `match_thresh`
