import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {DType} from '@tensorflow/tfjs-core/dist/types';
import {InputLayer, Layer, LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Initializer} from '@tensorflow/tfjs-layers/dist/initializers';
import {onnx} from 'onnx-proto';

import {ConstantCompat} from './compat/core';
import {getCommonConfig} from './layer_util';
import {OnnxModel} from './model';

export type StaticThis<T> = {
  new (model: OnnxModel): T
};

export abstract class OnnxNode {
  protected constructor(public model: OnnxModel) {};
  static from<T extends OnnxNode>(this: StaticThis<T>, model: OnnxModel): T {
    const that = new this(model);
    return that;
  }
  abstract getTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]): Layer;

  getTfjsLayerConfig(node: onnx.INodeProto, input?: SymbolicTensor[]):
      LayerConfig {
    return {};
  }

  getTfjsConfig(node: onnx.INodeProto, input?: SymbolicTensor[]): LayerConfig {
    const commonConfig = getCommonConfig(node);
    const layerConfig = this.getTfjsLayerConfig(node, input);
    return Object.assign({}, commonConfig, layerConfig);
  }

  prepareInput(input?: SymbolicTensor[]): SymbolicTensor[] {
    return input;
  }

  setup(node: onnx.INodeProto, input?: SymbolicTensor[]):
      [Layer, SymbolicTensor]|SymbolicTensor[] {
    const layer = this.getTfjsLayer(node, input);

    if (layer instanceof ConstantCompat || layer instanceof InputLayer) {
      const outputs = layer.inboundNodes[0].outputTensors;
      return [layer, outputs[0]];
    }

    return [layer, layer.apply(this.prepareInput(input)) as SymbolicTensor];
  }
}

export class WeightInitializer extends Initializer {
  constructor(protected weights: Tensor) {
    super();
  };
  apply(shape: number[], dtype?: DType): tf.Tensor<tf.Rank> {
    return this.weights;
  }
  getClassName(): string {
    return 'WeightInitializer';
  }
}
