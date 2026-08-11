# BitUNet Architecture Documentation

## 1. Overview
BitUNet is a highly optimized semantic segmentation model based on the standard UNet architecture. It is specifically designed for high-fidelity, real-time inference in resource-constrained environments (such as GPS-denied drone navigation). The core innovation of BitUNet is the application of the **BitNet b1.58 quantization scheme**, which quantizes the vast majority of network weights to ternary values `{-1, 0, 1}` and activations to 8-bit integers (INT8).

## 2. Architectural Design
The BitUNet architecture retains the classical Encoder-Bridge-Decoder structure of UNet with skip connections, but heavily modifies the convolution operations:

### 2.1. Hybrid Precision Strategy
To maintain representation capacity and geometric accuracy while maximizing compression, BitUNet employs a hybrid precision approach:
* **FP32 Interfaces**: The first convolution layer (input encoder) and the final convolution head (output) are kept in full floating-point precision (FP32). This ensures that the input image dynamics and output class probabilities are preserved without degradation.
* **Unquantized Normalization**: All `BatchNorm2d` layers remain in FP32 to stabilize training and maintain precise activation scaling.
* **Ternary Core Layers**: All intermediate feature extraction and upsampling layers (inside the `Down`, `Up`, and `DoubleConv` blocks) utilize a custom `BitConv2d` module instead of the standard `nn.Conv2d`.

### 2.2. BitConv2d Module
The `BitConv2d` layer acts as a drop-in replacement for standard 2D convolutions but operates with quantized tensors during the forward pass:
* **Weight Quantization**: Weights are quantized using a `RoundClip` function. Every weight is mapped to exactly `{-1, 0, 1}` by dividing by the mean absolute value of the weight tensor (the scale factor $\beta$) and rounding.
* **Activation Quantization**: Activations are quantized to INT8 using a per-sample absolute maximum scaling strategy, mapping values to the `[-128, 127]` range.
* **Straight-Through Estimator (STE)**: During training, non-differentiable rounding operations are bypassed in the backward pass using the `.detach()` trick, allowing gradients to flow to the latent FP32 weights.

### 2.3. Channel Compensation
Because extreme ternary quantization inherently reduces the representational capacity of a single filter, the BitUNet architecture slightly expands the network width. The number of base channels is increased from the standard **64 to 72**. This provides the network with more feature maps to compensate for the reduced capacity per parameter, ensuring high-fidelity segmentation without significantly impacting the overall compressed size.

## 3. Layer-by-Layer Structure
The network takes an input of shape `[Batch, 3, H, W]` and outputs `[Batch, 3, H, W]` corresponding to 3 classes (e.g., road, obstacle, background). The base channel count is denoted as $c=72$.

### Encoder
* **enc1**: The initial interface layer. Uses two standard float32 `Conv2d -> BatchNorm2d -> ReLU` operations. Transforms the 3-channel input into $c$ feature maps.
* **down1**: Halves the spatial resolution using a `MaxPool2d(2)`, followed by a ternary `DoubleConv` (which applies two `BitConv2d -> BatchNorm2d -> ReLU` blocks). Outputs $2c$ channels.
* **down2**: MaxPool2d(2) + `DoubleConv`. Outputs $4c$ channels.
* **down3**: MaxPool2d(2) + `DoubleConv`. Outputs $8c$ channels.

### Bridge
* **bridge**: Connects the encoder and decoder. Applies MaxPool2d(2) + `DoubleConv`. Outputs $16c$ channels at 1/16th of the original spatial resolution.

### Decoder
* **up3**: Upsamples the bridge output by a factor of 2 (bilinear). Concatenates it with the $8c$ skip connection from `down3`, resulting in $16c + 8c$ channels. Applies a ternary `DoubleConv` to output $8c$ channels.
* **up2**: Upsamples and concatenates with `down2`'s $4c$ skip connection. Applies `DoubleConv` to output $4c$ channels.
* **up1**: Upsamples and concatenates with `down1`'s $2c$ skip connection. Applies `DoubleConv` to output $2c$ channels.
* **up0**: Upsamples and concatenates with `enc1`'s $c$ skip connection. Applies `DoubleConv` to output $c$ channels at the original resolution.

### Output Head
* **head**: The final interface layer. Uses a single standard float32 `Conv2d` layer with kernel size 1 to map the $c$ feature maps down to the 3 target classes.

## 4. Model Size Reduction
The primary advantage of the BitUNet architecture is its dramatic reduction in memory footprint:
* Standard models use **32 bits (4 bytes)** per weight parameter.
* The `BitConv2d` layers in BitUNet represent their weights using only **1.58 bits** per parameter (since $\log_2(3) \approx 1.58$ to encode 3 states: -1, 0, 1).
* **Compression Ratio**: By keeping only the input/output layers in FP32 and pushing the bulk of the network into 1.58-bit ternary precision, the total model size is reduced significantly. As implemented, the float32 UNet is compressed heavily since the vast majority of parameters ($>90\%$) are represented in 1.58 bits rather than 32 bits, resulting in a model roughly 10x-15x smaller in memory.

## 5. Faster Inference Rate
BitUNet is not only smaller but significantly faster during inference, due to two main factors:
1. **Reduced Memory Bandwidth**: In memory-bound applications (like semantic segmentation on edge devices), loading weights from memory to the compute units is the primary bottleneck. The 1.58-bit weights drastically reduce memory bandwidth requirements, allowing the model to be loaded and executed much faster.
2. **Multiplication-Free Convolutions**: In standard neural networks, convolutions require thousands of expensive floating-point multiplications (MAC operations). Because BitUNet weights are constrained to `{-1, 0, 1}`, the dot products in convolution layers devolve into **pure integer additions and subtractions**. This eliminates the need for complex multipliers, allowing the network to be executed with much higher throughput and lower energy consumption on hardware that supports sparse or ternary operations (like modern FPGAs or specialized NPUs/GPUs).

## 6. Domain-Specific Additions: Path-Centric Inference
Beyond the core architecture, the BitUNet pipeline incorporates a specialized **Path-Centric Algorithm** designed for drone localization and navigation.
* It projects 3D waypoints from a planned path into the 2D image coordinates using camera intrinsics.
* It generates a Gaussian weight mask (a "corridor") centered on this projected path.
* The semantic segmentation probabilities (e.g., obstacle class) output by the BitUNet are weighted by this mask. This allows the system to trigger obstacle alerts only if an obstruction lies directly within the drone's intended flight corridor, minimizing false positives from irrelevant background structures.
