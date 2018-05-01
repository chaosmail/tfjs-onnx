import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {DType} from '@tensorflow/tfjs-core/dist/types';
import {Layer, LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Initializer} from '@tensorflow/tfjs-layers/dist/initializers';
import {onnx} from 'onnx-proto';

import {OnnxModel} from './base';
import * as util from './util';

export type StaticThis<T> = {
  new (model: OnnxModel): T
};

export abstract class OnnxNode {
  protected constructor(public model: OnnxModel) {};
  static from<T extends OnnxNode>(this: StaticThis<T>, model: OnnxModel): T {
    const that = new this(model);
    return that;
  }
  abstract getTfjsLayerConfig(node: onnx.INodeProto, input?: SymbolicTensor[]):
      LayerConfig;
  abstract getTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]): Layer;

  getTfjsConfig(node: onnx.INodeProto, input?: SymbolicTensor[]): LayerConfig {
    const commonConfig = util.getTfjsCommonConfig(node);
    const layerConfig = this.getTfjsLayerConfig(node, input);
    return Object.assign({}, commonConfig, layerConfig);
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
