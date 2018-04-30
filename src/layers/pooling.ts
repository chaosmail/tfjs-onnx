import * as tf from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Pooling1DLayerConfig, Pooling2DLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/pooling';
import {onnx} from 'onnx-proto';

import {OnnxNode} from '../node';
import {getNamedAttrs, parseAttrOrDefault} from '../util';

import {Conv} from './convolution';

export type PoolingLayerConfig = Pooling1DLayerConfig|Pooling2DLayerConfig;

export interface PoolNodeConfig {
  auto_pad?: onnx.AttributeProto;
  kernel_shape?: onnx.AttributeProto;
  pads?: onnx.AttributeProto;
  strides?: onnx.AttributeProto;
}

export abstract class Pool extends OnnxNode {
  getTfjsLayerConfig(node: onnx.INodeProto): PoolingLayerConfig {
    const conf = getNamedAttrs(node.attribute) as PoolNodeConfig;
    const poolSize = parseAttrOrDefault(conf.kernel_shape) as number;
    const strides = parseAttrOrDefault(conf.strides, 1) as number;
    const pads = parseAttrOrDefault(conf.pads, null);
    const autoPad = parseAttrOrDefault(conf.auto_pad, null);
    const padding = Conv.getTfjsPadding(pads, autoPad);

    return {
      poolSize: poolSize, strides: strides, padding: padding,
    }
  }

  static getPoolDim(node: onnx.INodeProto): number {
    const conf = getNamedAttrs(node.attribute) as PoolNodeConfig;
    return parseAttrOrDefault(conf.kernel_shape, []).length || 2;
  }
}

export class MaxPool extends Pool {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Pool.getPoolDim(node);
    const conf = this.getTfjsConfig(node) as PoolingLayerConfig;
    return dim == 1 ? tf.layers.maxPooling1d(conf as Pooling1DLayerConfig) :
                      tf.layers.maxPooling2d(conf as Pooling2DLayerConfig);
  }
}

export class AveragePool extends Pool {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Pool.getPoolDim(node);
    const conf = this.getTfjsConfig(node) as PoolingLayerConfig;
    return dim == 1 ? tf.layers.averagePooling1d(conf as Pooling1DLayerConfig) :
                      tf.layers.averagePooling2d(conf as Pooling2DLayerConfig);
  }
}

export class GlobalMaxPool extends Pool {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Pool.getPoolDim(node);
    const conf = this.getTfjsConfig(node) as PoolingLayerConfig;
    return dim == 1 ?
        tf.layers.globalMaxPooling1d(conf as Pooling1DLayerConfig) :
        tf.layers.globalMaxPooling2d(conf as Pooling2DLayerConfig);
  }
}

export class GlobalAveragePool extends Pool {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Pool.getPoolDim(node);
    const conf = this.getTfjsConfig(node) as PoolingLayerConfig;
    return dim == 1 ?
        tf.layers.globalAveragePooling1d(conf as Pooling1DLayerConfig) :
        tf.layers.globalAveragePooling2d(conf as Pooling2DLayerConfig);
  }
}
