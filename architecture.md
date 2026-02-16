## Architecture

This document describes the data flow architecture for the Remote Sensing ML pipeline. All diagrams use Mermaid to visualize how data moves through the system.

---

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph External["External Sources"]
        S2[Sentinel-2 Satellite]
        COH[Copernicus Open Access Hub]
        AWS[AWS Open Data Registry]
    end
    
    subgraph DataLayer["Data Layer"]
        RAW[Raw Imagery<br/>.SAFE format]
        TILE[Tiled Datasets<br/>256x256 patches]
        NORM[Normalized Data<br/>per-band scaling]
    end
    
    subgraph TrainingLayer["Training Layer"]
        DS[PyTorch Dataset]
        DL[DataLoader]
        MODEL[UNet Model]
        TRAIN[Training Loop<br/>AMP + Checkpointing]
    end
    
    subgraph ModelRegistry["Model Registry"]
        PT[PyTorch Weights<br/>.pth files]
        ONNX[ONNX Model<br/>optimized]
        META[Model Metadata<br/>version, metrics]
    end
    
    subgraph ServingLayer["Serving Layer"]
        API[FastAPI Service]
        INF[Inference Engine<br/>ONNX Runtime]
        CACHE[Model Cache<br/>lazy loading]
    end
    
    subgraph MonitoringLayer["Monitoring Layer"]
        PROM[Prometheus Metrics]
        LOG[Structured Logs]
        HEALTH[Health Checks]
    end
    
    subgraph Clients["Clients"]
        WEB[Web Application]
        MOBILE[Mobile App]
        BATCH[Batch Processing]
    end
    
    S2 --> COH
    S2 --> AWS
    COH --> RAW
    AWS --> RAW
    RAW --> TILE
    TILE --> NORM
    NORM --> DS
    DS --> DL
    DL --> TRAIN
    MODEL --> TRAIN
    TRAIN --> PT
    PT --> ONNX
    ONNX --> META
    ONNX --> INF
    INF --> API
    API --> CACHE
    API --> PROM
    API --> LOG
    API --> HEALTH
    API --> WEB
    API --> MOBILE
    API --> BATCH
    
    style External fill:#e1f5ff
    style DataLayer fill:#e8f5e9
    style TrainingLayer fill:#fff3e0
    style ModelRegistry fill:#f3e5f5
    style ServingLayer fill:#ffebee
    style MonitoringLayer fill:#e0f2f1
    style Clients fill:#fce4ec
```

---

## Data Ingestion Flow

```mermaid
flowchart LR
    subgraph Source["Data Source"]
        S2[Sentinel-2 Scene<br/>10000x10000px<br/>13 bands]
    end
    
    subgraph Download["Download"]
        SAFE[.SAFE Container<br/>JP2 format]
        BAND[Individual Bands<br/>B2-B12]
    end
    
    subgraph Preprocess["Preprocessing"]
        READ[Read with Rasterio<br/>numpy arrays]
        STACK[Stack Bands<br/>H x W x C]
        TILE[Tiling<br/>256x256 patches]
        NORM[Normalization<br/>per-band scaling]
    end
    
    subgraph Storage["Storage"]
        RAW[Raw Tiles<br/>float32]
        MASK[Ground Truth Masks<br/>uint8]
        META[Metadata JSON<br/>geospatial info]
    end
    
    subgraph Consumption["Consumption"]
        DS[Dataset Class<br/>__getitem__]
        AUG[Augmentation<br/>flip, rotate, noise]
        TENSOR[PyTorch Tensor<br/>B x C x H x W]
    end
    
    S2 -->|ESA API| SAFE
    SAFE -->|Extract| BAND
    BAND --> READ
    READ --> STACK
    STACK -->|Sliding Window| TILE
    TILE -->|z-score or min-max| NORM
    NORM --> RAW
    NORM --> MASK
    RAW --> META
    RAW --> DS
    MASK --> DS
    META --> DS
    DS -->|On-the-fly| AUG
    AUG --> TENSOR
    
    style Source fill:#e3f2fd
    style Download fill:#e8f5e9
    style Preprocess fill:#fff3e0
    style Storage fill:#f3e5f5
    style Consumption fill:#ffebee
```

**Key Data Transformations**:
- **JP2 → numpy**: Lossless decompression
- **H x W x C → C x H x W**: Channel-first for PyTorch
- **uint16 → float32**: Normalization requires float
- **256x256 patches**: Memory-efficient training

---

## Training Pipeline Flow

```mermaid
flowchart TB
    subgraph Input["Training Input"]
        TILE[Image Tiles<br/>B x C x H x W]
        MASK[Ground Truth<br/>B x H x W]
    end
    
    subgraph DataPipeline["Data Pipeline"]
        DL[DataLoader<br/>batch_size=16]
        GPU[(Transfer to<br/>GPU Memory)]
    end
    
    subgraph ForwardPass["Forward Pass"]
        ENC[Encoder<br/>5 levels<br/>64→128→256→512→1024]
        BOT[Bottleneck<br/>16x16x1024]
        DEC[Decoder<br/>5 levels<br/>1024→512→256→128→64]
        SKIP[Skip Connections<br/>concatenate]
        OUT[Output Conv<br/>1x1 kernel<br/>num_classes]
    end
    
    subgraph LossComputation["Loss Computation"]
        PRED[Predictions<br/>B x num_classes x H x W]
        LOSS_FN[Dice + CE Loss]
        LOSS[Scalar Loss]
    end
    
    subgraph BackwardPass["Backward Pass"]
        AMP[AMP Gradient<br/>Scaling]
        BACK[loss.backward]
        OPT[Optimizer Step<br/>AdamW]
        SCHED[LR Scheduler]
    end
    
    subgraph Checkpointing["Checkpointing"]
        SAVE[Save Checkpoint<br/>every N epochs]
        BEST[Save Best<br/>val IoU improved]
        DICT[Checkpoint Dict<br/>model + optimizer + scheduler]
    end
    
    subgraph Validation["Validation"]
        VAL[Validation Loop<br/>no_grad]
        METRICS[Calculate IoU<br/>Dice Score]
        LOG[Log Metrics<br/>TensorBoard/W&B]
    end
    
    TILE --> DL
    MASK --> DL
    DL --> GPU
    GPU --> ENC
    ENC --> BOT
    BOT --> DEC
    ENC -.->|skip| SKIP
    DEC --> SKIP
    SKIP --> OUT
    OUT --> PRED
    PRED --> LOSS_FN
    MASK --> LOSS_FN
    LOSS_FN --> LOSS
    LOSS --> AMP
    AMP --> BACK
    BACK --> OPT
    OPT --> SCHED
    SCHED --> SAVE
    SCHED --> BEST
    SAVE --> DICT
    BEST --> DICT
    PRED -->|eval mode| VAL
    MASK --> VAL
    VAL --> METRICS
    METRICS --> LOG
    
    style Input fill:#e3f2fd
    style DataPipeline fill:#e8f5e9
    style ForwardPass fill:#fff3e0
    style LossComputation fill:#ffccbc
    style BackwardPass fill:#f8bbd9
    style Checkpointing fill:#d1c4e9
    style Validation fill:#b2dfdb
```

**Data Flow Details**:
- **Batch size**: 16-32 depending on GPU memory
- **AMP**: Automatic Mixed Precision (FP16) for 2x speedup
- **Gradient accumulation**: For larger effective batch size
- **Checkpoint frequency**: Every epoch + when validation improves

---

## Model Export & Optimization Flow

```mermaid
flowchart LR
    subgraph SourceModel["Source Model"]
        PT[PyTorch Model<br/>model.pth]
        EVAL[eval mode]
        DUMMY[Dummy Input<br/>B x C x H x W]
    end
    
    subgraph Export["Export Process"]
        TRACE[ONNX Tracing<br/>torch.onnx.export]
        GRAPH[ONNX Graph<br/>Protobuf format]
        META[Metadata<br/>opset_version=11<br/>dynamic_axes]
    end
    
    subgraph Verification["Verification"]
        ORT[ONNX Runtime<br/>InferenceSession]
        COMPARE[Output Comparison<br/>PyTorch vs ONNX]
        TOL[Tolerance Check<br/>rtol=1e-3]
    end
    
    subgraph Optimization["Optimization"]
        OPT[Graph Optimization<br/>constant folding<br/>fusion]
        QUANT[Quantization<br/>FP32 → INT8]
        TENSORRT[TensorRT<br/>GPU optimization]
    end
    
    subgraph Deployment["Deployment"]
        FINAL[Optimized Model<br/>ready for serving]
        REGISTRY[Model Registry<br/>versioned artifacts]
    end
    
    PT --> EVAL
    EVAL --> DUMMY
    DUMMY --> TRACE
    TRACE --> GRAPH
    TRACE --> META
    GRAPH --> ORT
    DUMMY --> COMPARE
    ORT --> COMPARE
    COMPARE --> TOL
    TOL --> OPT
    OPT --> QUANT
    OPT --> TENSORRT
    QUANT --> FINAL
    TENSORRT --> FINAL
    FINAL --> REGISTRY
    
    style SourceModel fill:#e3f2fd
    style Export fill:#e8f5e9
    style Verification fill:#fff3e0
    style Optimization fill:#ffccbc
    style Deployment fill:#d1c4e9
```

**Export Details**:
- **Dynamic axes**: Allow variable batch sizes
- **Opset version**: 11 for good compatibility
- **Verification**: Numerical tolerance 1e-3 acceptable
- **Optimization**: ~1.5-3x speedup, 4x smaller with quantization

---

## Inference API Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Gateway
    participant Val as Validation Layer
    participant Cache as Model Cache
    participant Inf as Inference Engine
    participant Metrics as Prometheus
    
    Client->>API: POST /predict<br/>{image: base64}
    activate API
    
    API->>Metrics: Increment request counter
    API->>Val: Validate request format
    
    alt Invalid Request
        Val-->>API: ValidationError (422)
        API-->>Client: 422 Unprocessable Entity
    else Valid Request
        Val->>API: Parsed image tensor
        
        API->>Cache: Get model instance
        
        alt Model not loaded
            Cache->>Inf: Load ONNX model
            Inf-->>Cache: Model instance
        end
        
        Cache-->>API: Model ready
        
        API->>Inf: Run inference<br/>input tensor
        activate Inf
        
        Inf->>Inf: Preprocess<br/>resize, normalize
        Inf->>Inf: ONNX Runtime<br/>session.run()
        Inf->>Inf: Postprocess<br/>argmax, threshold
        
        Inf-->>API: predictions<br/>mask + confidence
        deactivate Inf
        
        API->>Metrics: Record latency<br/>histogram
        API->>Metrics: Update gauges
        
        API->>API: Format response<br/>JSON
        
        API-->>Client: 200 OK<br/>{mask, confidence}
    end
    
    deactivate API
    
    Note over Client,Metrics: Typical latency: 50-200ms<br/>Throughput: 50-100 req/sec
```

**Request Flow**:
1. Client sends HTTP POST with image
2. API validates format and size
3. Model loaded once (lazy initialization)
4. Preprocessing (resize to 256x256, normalize)
5. ONNX Runtime inference
6. Postprocessing (argmax, confidence scores)
7. Metrics recorded
8. JSON response returned

---

## End-to-End Data Flow Summary

```mermaid
flowchart LR
    subgraph Data["Data Acquisition"]
        S2[Sentinel-2]
        DOWNLOAD[Download]
        PREP[Preprocess]
    end
    
    subgraph Train["Model Training"]
        DATALOADER[DataLoader]
        UNET[UNet]
        LOSS[Loss Function]
        OPT[Optimizer]
    end
    
    subgraph Deploy["Deployment"]
        EXPORT[ONNX Export]
        API[FastAPI]
        CACHE[Model Cache]
    end
    
    subgraph Serve["Serving"]
        REQUEST[HTTP Request]
        PREDICT[Inference]
        RESPONSE[JSON Response]
    end
    
    subgraph Monitor["Monitoring"]
        METRICS[Prometheus]
        LOGS[Logging]
        ALERTS[Alerts]
    end
    
    S2 --> DOWNLOAD
    DOWNLOAD --> PREP
    PREP --> DATALOADER
    DATALOADER --> UNET
    UNET --> LOSS
    LOSS --> OPT
    OPT --> EXPORT
    EXPORT --> API
    API --> CACHE
    CACHE --> PREDICT
    REQUEST --> PREDICT
    PREDICT --> RESPONSE
    PREDICT --> METRICS
    PREDICT --> LOGS
    METRICS --> ALERTS
    LOGS --> ALERTS
    
    style Data fill:#bbdefb
    style Train fill:#c8e6c9
    style Deploy fill:#fff9c4
    style Serve fill:#ffccbc
    style Monitor fill:#e1bee7
```

**Summary**:
1. **Data**: Satellite → Preprocessed tiles
2. **Train**: UNet trained with AMP + checkpointing
3. **Deploy**: Export to ONNX, serve with FastAPI
4. **Serve**: REST API for real-time predictions
5. **Monitor**: Metrics, logs, and alerting

---

## Notes

- All diagrams are interactive and can be viewed with Mermaid-supported viewers
- Data flows are designed to be modular and replaceable
- Each stage has clear inputs and outputs for testing
- Monitoring is integrated at every layer for observability
