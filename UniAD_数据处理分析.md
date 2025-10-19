# UniAD 数据处理流程与DataLoader内容格式分析

## 一、整体数据处理架构

UniAD框架采用了典型的MMDetection3D数据处理架构，主要包含以下几个层次：

```
Dataset (NuScenesE2EDataset) 
    ↓
Pipeline (数据增强与预处理)
    ↓
Collate (批处理整合)
    ↓
DataLoader
```

---

## 二、数据集类 (NuScenesE2EDataset)

### 2.1 核心功能

`NuScenesE2EDataset` 继承自 `NuScenesDataset`，是UniAD的核心数据集类，位于：
- `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`

### 2.2 关键初始化参数

```python
def __init__(self,
    queue_length=4,           # 时序队列长度（历史帧数量）
    bev_size=(200, 200),      # BEV特征图大小
    patch_size=(102.4, 102.4), # 地图patch大小（米）
    canvas_size=(200, 200),    # 地图画布大小
    predict_steps=12,          # 预测未来轨迹步数
    planning_steps=6,          # 规划步数
    past_steps=4,              # 过去轨迹步数
    fut_steps=4,               # 未来轨迹步数
    # Occupancy相关
    occ_receptive_field=3,     # 占用网格感受野（过去+当前）
    occ_n_future=4,            # 占用网格未来帧数
    ...
)
```

### 2.3 核心数据获取流程

#### 训练模式：`prepare_train_data(index)`

1. **时序队列构建**：
   - 获取当前帧和前 `queue_length-1` 帧的数据
   - 确保所有帧来自同一场景 (scene_token)
   
2. **单帧数据获取** (`get_data_info`):
   - 基础信息：sample_idx, timestamp, scene_token
   - 传感器数据路径：图像路径、雷达路径
   - 坐标变换：lidar2img, lidar2cam, l2g (lidar-to-global)
   - 地图信息：矢量化地图、栅格化地图
   - 标注信息：3D框、轨迹、planning标签

3. **Pipeline处理**：每帧数据都会经过预处理pipeline

4. **时序数据融合** (`union2one`):
   - 将多帧数据打包成一个sample
   - 计算相对位姿变化（用于时序BEV对齐）

#### 测试模式：`prepare_test_data(index)`

- 只获取单个时刻的数据
- 使用测试pipeline（不包含数据增强）

---

## 三、数据Pipeline处理

### 3.1 训练Pipeline

完整的训练数据处理流程（从 `base_e2e.py` 配置）：

```python
train_pipeline = [
    # 1. 加载图像数据
    dict(type="LoadMultiViewImageFromFilesInCeph", 
         to_float32=True),
    
    # 2. 图像增强
    dict(type="PhotoMetricDistortionMultiViewImage"),
    
    # 3. 加载3D标注
    dict(type="LoadAnnotations3D_E2E",
         with_bbox_3d=True,          # 3D检测框
         with_label_3d=True,         # 类别标签
         with_future_anns=True,      # 未来帧标注（用于Occ）
         with_ins_inds_3d=True,      # 实例ID
         ins_inds_add_1=True),       # 实例ID从1开始
    
    # 4. 生成Occupancy和Flow标签
    dict(type='GenerateOccFlowLabels', 
         grid_conf=occflow_grid_conf,
         only_vehicle=True),
    
    # 5. 过滤与筛选
    dict(type="ObjectRangeFilterTrack", 
         point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", 
         classes=class_names),
    
    # 6. 图像归一化与填充
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    
    # 7. 格式化
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    
    # 8. 收集最终数据
    dict(type="CustomCollect3D", keys=[...]),
]
```

### 3.2 测试Pipeline

测试时的简化pipeline：
- 移除数据增强
- 使用MultiScaleFlipAug3D支持多尺度测试
- 保留标注加载（用于评估）

---

## 四、DataLoader中的数据格式

经过完整pipeline处理后，DataLoader返回的单个batch包含以下内容：

### 4.1 图像数据

```python
'img': DataContainer
    # Shape: (batch_size, queue_length, num_cams, C, H, W)
    # queue_length=4: 历史3帧 + 当前帧
    # num_cams=6: 6个环视相机
    # C=3: RGB通道
    # H, W: 填充后的图像尺寸（能被32整除）
```

### 4.2 元数据 (img_metas)

```python
'img_metas': DataContainer (cpu_only=True)
    # 字典格式，key为时间步索引 {0, 1, 2, 3}
    {
        0: {  # 最早的历史帧
            'sample_idx': str,
            'timestamp': float,
            'scene_token': str,
            'can_bus': np.ndarray,  # 车辆状态 [x,y,z, quat, vel, acc, angle]
            'lidar2img': list,      # 6个相机的投影矩阵
            'lidar2cam': list,
            'cam_intrinsic': list,
            'prev_bev': bool,       # 是否有前一帧BEV
            ...
        },
        1: {...},  # 历史帧
        2: {...},  # 历史帧
        3: {...},  # 当前帧
    }
```

### 4.3 检测相关标注

```python
# 3D边界框（每个时间步）
'gt_bboxes_3d': DataContainer (cpu_only=True)
    # List[LiDARInstance3DBoxes], length=queue_length
    # 每个元素shape: (num_objects, 9)  # [x,y,z,w,l,h,yaw,vx,vy]

# 类别标签（每个时间步）
'gt_labels_3d': DataContainer
    # List[Tensor], length=queue_length
    # 每个元素shape: (num_objects,)

# 实例ID（每个时间步）
'gt_inds': DataContainer
    # List[Tensor], length=queue_length
    # 用于跨帧tracking

# 自车（SDC）边界框
'gt_sdc_bbox': DataContainer (cpu_only=True)
    # List[LiDARInstance3DBoxes], length=queue_length

'gt_sdc_label': DataContainer
    # List[Tensor], length=queue_length
```

### 4.4 轨迹预测标注

```python
# 历史轨迹（每个时间步的物体）
'gt_past_traj': DataContainer
    # List[Tensor], length=queue_length
    # 每个元素shape: (num_objects, past_steps, 2)  # [x, y]

'gt_past_traj_mask': DataContainer
    # List[Tensor], length=queue_length
    # 每个元素shape: (num_objects, past_steps)

# 未来轨迹（仅最后一帧）
'gt_fut_traj': DataContainer
    # Tensor, shape: (num_objects, fut_steps, 2)

'gt_fut_traj_mask': DataContainer
    # Tensor, shape: (num_objects, fut_steps)

# 自车未来轨迹
'gt_sdc_fut_traj': DataContainer
    # Tensor, shape: (fut_steps, 2)

'gt_sdc_fut_traj_mask': DataContainer
    # Tensor, shape: (fut_steps,)
```

### 4.5 地图分割标注

```python
'gt_lane_labels': Tensor
    # Shape: (num_map_instances,)
    # 地图元素类别（车道线、道路边界等）

'gt_lane_bboxes': Tensor
    # Shape: (num_map_instances, 4)  # [x1, y1, x2, y2]

'gt_lane_masks': Tensor
    # Shape: (num_map_instances, map_H, map_W)
    # 地图元素的实例mask
```

### 4.6 Occupancy & Flow标注

```python
'gt_segmentation': Tensor
    # Shape: (n_future+1, Z, X, Y)
    # 语义占用网格，n_future=4

'gt_instance': Tensor
    # Shape: (n_future+1, Z, X, Y)
    # 实例占用网格

'gt_centerness': Tensor
    # Shape: (n_future+1, 1, X, Y)
    # 实例中心性

'gt_offset': Tensor
    # Shape: (n_future+1, 2, X, Y)
    # 到实例中心的偏移

'gt_flow': Tensor
    # Shape: (n_future, 2, X, Y)
    # 前向光流

'gt_backward_flow': Tensor
    # Shape: (n_future, 2, X, Y)
    # 后向光流

'gt_occ_has_invalid_frame': bool
    # 是否包含无效帧

'gt_occ_img_is_valid': Tensor
    # Shape: (total_frames,)
    # 每帧是否有效
```

### 4.7 规划标注

```python
'gt_future_boxes': DataContainer (cpu_only=True)
    # List[LiDARInstance3DBoxes]
    # 未来时刻的物体框（用于规划）

'gt_future_labels': DataContainer
    # List[Tensor]
    # 未来时刻的物体类别

'sdc_planning': Tensor
    # Shape: (planning_steps, planning_dim)
    # 自车规划轨迹标签

'sdc_planning_mask': Tensor
    # Shape: (planning_steps,)
    # 规划轨迹mask

'command': Tensor
    # 高层指令（左转、右转、直行等）
```

### 4.8 坐标变换

```python
'l2g_r_mat': DataContainer
    # List[Tensor], length=queue_length
    # Lidar到全局坐标系的旋转矩阵
    # 每个元素shape: (3, 3)

'l2g_t': DataContainer
    # List[Tensor], length=queue_length
    # Lidar到全局坐标系的平移向量
    # 每个元素shape: (3,)

'timestamp': DataContainer
    # List[Tensor], length=queue_length
    # 每帧的时间戳
```

---

## 五、数据处理的关键特点

### 5.1 时序建模

- **Queue机制**：维护长度为4的时序队列（3个历史帧+1个当前帧）
- **相对位姿**：计算帧间相对位姿用于时序BEV特征对齐
- **时序一致性**：确保所有帧来自同一场景

### 5.2 多任务标注

UniAD是端到端的多任务框架，DataLoader包含：
1. **感知任务**：3D检测、Tracking、地图分割
2. **预测任务**：多智能体轨迹预测
3. **占用任务**：语义占用网格、占用流
4. **规划任务**：自车规划轨迹

### 5.3 坐标系统

- **Lidar坐标系**：主要工作坐标系
- **全局坐标系**：用于跨帧tracking和轨迹预测
- **BEV坐标系**：用于地图和占用网格

### 5.4 DataContainer封装

使用MMDetection的`DataContainer`类封装数据：
- `cpu_only=True`: 大型数据（如3D框）保留在CPU
- `stack=True`: 支持batch堆叠
- 自动处理不同设备间的数据传输

---

## 六、Batch构建流程

```
1. Sampler采样indices
   ↓
2. Dataset.__getitem__(idx)
   ↓
3. prepare_train_data(idx)
   ↓
4. 构建时序队列（4帧）
   ↓
5. 每帧执行pipeline处理
   ↓
6. union2one融合多帧
   ↓
7. collate_fn批处理整合
   ↓
8. 返回batch数据
```

---

## 七、关键代码位置

| 功能 | 文件路径 |
|------|---------|
| Dataset类 | `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py` |
| DataLoader构建 | `projects/mmdet3d_plugin/datasets/builder.py` |
| Pipeline定义 | `projects/configs/stage2_e2e/base_e2e.py` |
| 图像加载 | `projects/mmdet3d_plugin/datasets/pipelines/loading.py` |
| 数据格式化 | `projects/mmdet3d_plugin/datasets/pipelines/formating.py` |
| Occ标签生成 | `projects/mmdet3d_plugin/datasets/pipelines/occflow_label.py` |

---

## 八、总结

UniAD的DataLoader设计的核心特点：

1. **时序感知**：通过queue机制整合多帧历史信息
2. **多模态融合**：6个相机图像+坐标变换+地图信息
3. **多任务支持**：一次加载支持检测、跟踪、预测、占用、规划等多个任务
4. **高效设计**：使用DataContainer优化GPU/CPU数据传输
5. **灵活配置**：通过pipeline机制支持不同的数据增强策略

最终的batch数据包含了完整的端到端自动驾驶所需的所有输入和监督信号。
