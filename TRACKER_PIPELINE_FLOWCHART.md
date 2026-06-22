# Pipeline Flowchart: ByteTrack vs BoT-SORT

File này trình bày sự khác nhau giữa `ByteTrack` và `BoT-SORT` theo thứ tự: flowchart trước, giải thích sau.

## 1. Sự Khác Nhau Cốt Lõi

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "24px",
    "primaryTextColor": "#111827",
    "primaryColor": "#F8FAFC",
    "primaryBorderColor": "#334155",
    "lineColor": "#374151"
  },
  "flowchart": {
    "nodeSpacing": 38,
    "rankSpacing": 45
  }
}}%%
flowchart LR
    A["VIDEO FRAME"] --> B["YOLO DETECTION"]
    B --> C["BOXES + SCORES"]

    C --> BT1
    C --> BS1

    subgraph BT["BYTETRACK"]
        direction TB
        BT1["Uses box position"]
        BT2["Uses detection score"]
        BT3["Uses low-score boxes to recover tracks"]
        BT4["No appearance ReID by default"]
        BT5["Result: faster FPS"]
        BT6["Weakness: easier ID switch"]
        BT1 --> BT2 --> BT3 --> BT4 --> BT5 --> BT6
    end

    subgraph BS["BOT-SORT"]
        direction TB
        BS1["Uses box position"]
        BS2["Uses detection score"]
        BS3["Adds camera motion compensation"]
        BS4["Adds appearance ReID"]
        BS5["Result: more stable ID"]
        BS6["Weakness: slower FPS"]
        BS1 --> BS2 --> BS3 --> BS4 --> BS5 --> BS6
    end

    classDef byte fill:#ECFDF5,stroke:#059669,color:#111827;
    classDef bot fill:#FFF7ED,stroke:#EA580C,color:#111827;
    class BT1,BT2,BT3,BT4,BT5,BT6 byte;
    class BS1,BS2,BS3,BS4,BS5,BS6 bot;
```

## 2. Pipeline Đặt Cạnh Nhau

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "24px",
    "primaryTextColor": "#111827",
    "primaryColor": "#F8FAFC",
    "primaryBorderColor": "#334155",
    "lineColor": "#374151"
  },
  "flowchart": {
    "nodeSpacing": 38,
    "rankSpacing": 45
  }
}}%%
flowchart LR
    A["VIDEO FRAME"] --> B["YOLO DETECTION"]
    B --> C["BOXES + SCORES"]

    C --> BT1
    C --> BS1

    subgraph BT["BYTETRACK PIPELINE"]
        direction TB
        BT1["1. Split high / low score boxes"]
        BT2["2. Kalman predict"]
        BT3["3. Match high-score boxes"]
        BT4["4. Recover with low-score boxes"]
        BT5["5. Update / create IDs"]
        BT6["OUTPUT: fast tracking"]
        BT1 --> BT2 --> BT3 --> BT4 --> BT5 --> BT6
    end

    subgraph BS["BOT-SORT PIPELINE"]
        direction TB
        BS1["1. Kalman predict"]
        BS2["2. GMC camera compensation"]
        BS3["3. IoU + score distance"]
        BS4["4. ReID appearance matching"]
        BS5["5. Fuse distances + match"]
        BS6["OUTPUT: stable IDs"]
        BS1 --> BS2 --> BS3 --> BS4 --> BS5 --> BS6
    end

    classDef byte fill:#ECFDF5,stroke:#059669,color:#111827;
    classDef bot fill:#FFF7ED,stroke:#EA580C,color:#111827;
    class BT1,BT2,BT3,BT4,BT5,BT6 byte;
    class BS1,BS2,BS3,BS4,BS5,BS6 bot;
```

## 3. Giải Thích Khác Nhau

| Tiêu chí | ByteTrack | BoT-SORT |
|---|---|---|
| Thông tin chính dùng để tracking | Vị trí box, IoU, confidence score | Vị trí box, IoU, confidence score, camera motion, appearance |
| Cách giữ ID | Ghép track cũ với detection mới bằng chuyển động và overlap box | Ghép bằng chuyển động + overlap box + bù chuyển động camera + đặc trưng ngoại hình |
| Low-score detection | Dùng rất mạnh để cứu track bị yếu confidence | Có dùng logic kế thừa từ ByteTrack, nhưng thêm các cue khác |
| ReID appearance | Không phải trọng tâm chính | Là điểm mạnh quan trọng khi bật `with_reid` |
| Camera motion compensation | Không phải phần chính | Có GMC, hữu ích khi camera rung hoặc di chuyển |
| Tốc độ FPS | Thường nhanh hơn | Thường chậm hơn |
| Độ ổn định ID | Tốt trong cảnh đơn giản, ít che khuất | Tốt hơn trong cảnh đông người, che khuất, người cắt nhau |
| Rủi ro chính | Dễ đổi ID hơn khi hai người gần nhau hoặc bị che | Tốn tài nguyên hơn, giảm FPS |

## 4. Hiểu Ngắn Gọn

`ByteTrack` cố gắng giữ ID bằng cách tận dụng cả detection mạnh và detection yếu. Khi một người bị che khuất nhẹ hoặc confidence giảm, detection đó có thể rơi xuống nhóm low-score. ByteTrack vẫn dùng nhóm này để nối lại track cũ, vì vậy thuật toán nhẹ và nhanh.

`BoT-SORT` mở rộng ý tưởng tracking bằng box của ByteTrack, nhưng thêm hai nguồn thông tin quan trọng: `GMC` để bù chuyển động camera và `ReID` để so sánh ngoại hình người. Vì vậy BoT-SORT thường giữ ID tốt hơn trong cảnh khó, nhưng phải trả giá bằng tốc độ thấp hơn.

## 5. Khi Nào Chọn Thuật Toán Nào

- Chọn `ByteTrack` nếu ưu tiên FPS, realtime, chạy trên Jetson/edge, và ID switch chỉ tăng nhẹ.
- Chọn `BoT-SORT` nếu cảnh có nhiều người giao nhau, che khuất lâu, camera rung/di chuyển, hoặc yêu cầu giữ ID quan trọng hơn tốc độ.
- Với kết quả bạn quan sát được, nếu `ByteTrack` chỉ kém `BoT-SORT` một chút về ID nhưng FPS tăng gần gấp đôi, `ByteTrack` là lựa chọn thực dụng hơn cho chạy thật.
