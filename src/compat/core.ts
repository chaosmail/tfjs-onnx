import {Shape} from '@tensorflow/tfjs';
import {Tensor} from '@tensorflow/tfjs-core/dist';
import {DType, Rank} from '@tensorflow/tfjs-core/dist/types';
import {InputLayer, Layer, LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Reshape, ReshapeLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/core';
import * as generic_utils from '@tensorflow/tfjs-layers/dist/utils/generic_utils';


export interface ConstantLayerConfig extends LayerConfig {
  value: Tensor, sparse?: boolean, inputShape?: number[]
}

export class ConstantLayer extends InputLayer {
  static readonly className = 'ConstantLayer';
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
    return ConstantLayer.className;
  }
}

export class ReshapeLayer extends Reshape {
  static className = 'ReshapeLayer';
  private reTargetShape: Shape;

  constructor(config: ReshapeLayerConfig) {
    super(config);
    this.reTargetShape = config.targetShape;

    // Make sure that all unknown dimensions are represented as `null`.
    for (let i = 0; i < this.reTargetShape.length; ++i) {
      if (this.reIsUnknown(this.reTargetShape[i])) {
        this.reTargetShape[i] = null;
      }
    }
  }

  private reIsUnknown(dim: number): boolean {
    return dim < 0 || dim == null;
  }

  computeOutputShape(inputShape: Shape): Shape {
    return this.reTargetShape;
  }

  call(inputs: Tensor|Tensor[]): Tensor|Tensor[] {
    const input = generic_utils.getExactlyOneTensor(inputs);
    return input.reshape(this.reTargetShape);
  }
}

export class MatMulLayer extends Layer {
  static className = 'MatMulLayer';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  computeOutputShape(inputShape: Shape[]): Shape {
    // TODO add checks
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

    return b.matMul(a).expandDims(0);
  }

  getClassName(): string {
    return MatMulLayer.className;
  }
}
