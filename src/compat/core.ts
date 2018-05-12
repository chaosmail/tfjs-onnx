import {Shape} from '@tensorflow/tfjs';
import {Tensor} from '@tensorflow/tfjs-core/dist';
import {DType, Rank} from '@tensorflow/tfjs-core/dist/types';
import {InputLayer, Layer, LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface ConstantLayerConfig extends LayerConfig {
  value: Tensor, sparse?: boolean, inputShape?: number[]
}

export class ConstantCompat extends InputLayer {
  static readonly className = 'ConstantCompat';
  public value: Tensor;

  constructor(config: ConstantLayerConfig) {
    super({
      name: config.name,
      dtype: config.value.dtype as DType,
      inputShape: config.inputShape,
      sparse: config.sparse
    });

    this.value = config.value;
  }

  call(inputs: Tensor|Tensor[]): Tensor|Tensor[] {
    return this.value;
  }

  getClassName(): string {
    return ConstantCompat.className;
  }
}

// TODO port back to tfjs
export class MatMulCompat extends Layer {
  static className = 'MatMulCompat';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  computeOutputShape(inputShape: Shape[]): Shape {
    // TODO add checks for computing the output shape
    const aShape = inputShape[0];
    const bShape = inputShape[1];
    return aShape.slice(0, 1).concat(bShape[1], aShape[2]);
  }

  call(inputs: Tensor[]): Tensor|Tensor[] {
    if (inputs.length !== 2) {
      throw new Error(`Layer 'MatMul' requires 2 inputs`);
    }

    const a = inputs[0].squeeze([0]) as Tensor<Rank.R2>;
    const b = inputs[1].squeeze([0]) as Tensor<Rank.R2>;

    // both tensors are transposed, hence we change the order
    return b.matMul(a).expandDims(0);
  }

  getClassName(): string {
    return MatMulCompat.className;
  }
}
