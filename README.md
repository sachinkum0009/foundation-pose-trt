# foundation-pose-trt
TensorRT Implementation for FoundationPose

## Architecture

> Sequence Diagram

```mermaid
graph LR
  A[Load Model] --> B[TensorRT Engine]
  C[Input Image] --> B
  B --> D[Output Tensor]
```

> Class Diagram

```mermaid
classDiagram
    class TRTEngine {
        - IExecutionContext context
        - ICudaEngine engine
        - List~void*~ buffers
        - cudaStream_t stream
        + TRTEngine(String modelPath)
        + infer(List inputs) List outputs
        - allocateBuffers()
    }

    class PreProcessor {
        + prepareImage(Mat rgb, Mat depth)
        + normalize(Tensor input)
        + createCrops(Rect roi)
    }

    class FoundationPose {
        - TRTEngine encoder
        - TRTEngine decoder
        - PreProcessor processor
        + init(String enginePath)
        + estimatePose(Mat rgb, Mat depth) Pose
        - postProcess(Tensor output)
    }

    class Pose {
        + Matrix4x4 transform
        + float confidence
        + getRotation()
        + getTranslation()
    }

    FoundationPose *-- TRTEngine : composes
    FoundationPose *-- PreProcessor : uses
    FoundationPose ..> Pose : produces
```

## Acknowledgement

- https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt
