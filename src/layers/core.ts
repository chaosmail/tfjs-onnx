import * as tf from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {DenseLayerConfig, DropoutLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/core';
import {onnx} from 'onnx-proto';

import {OnnxNode, WeightInitializer} from '../node';
import {getNamedAttrs, parseAttrOrDefault} from '../util';

export interface FCNodeConfig {
  axis?: onnx.AttributeProto;
  axis_w?: onnx.AttributeProto;
}

export class Dense extends OnnxNode {
  getTfjsLayerConfig(node: onnx.INodeProto): DenseLayerConfig {
    const w = node.input[1];
    const b = node.input[2];
    const weightShape = this.model.blobShapes[w];
    const units = weightShape[0];
    const kernel = this.model.blobValues[w];
    const bias = this.model.blobValues[b];

    return {
      units: units, kernelInitializer: new WeightInitializer(kernel),
          biasInitializer: new WeightInitializer(bias)
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node) as DenseLayerConfig;
    return tf.layers.dense(conf)
  }
}

export interface DropoutNodeConfig {
  is_test?: onnx.AttributeProto;
  ratio?: onnx.AttributeProto;
}

export class Dropout extends OnnxNode {
  getTfjsLayerConfig(node: onnx.INodeProto): DropoutLayerConfig {
    const conf = getNamedAttrs(node.attribute) as DropoutNodeConfig;
    const ratio = parseAttrOrDefault(conf.ratio, 0) as number;
    return {
      rate: ratio
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node) as DropoutLayerConfig;
    return tf.layers.dropout(conf);
  }
}
