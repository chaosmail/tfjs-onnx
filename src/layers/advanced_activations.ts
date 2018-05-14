import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {SoftmaxLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/advanced_activations';
import {onnx} from 'onnx-proto';

import {OnnxNode} from '../node';
import {parseAttrOrDefault, parseAxis} from '../onnx_util';
import {getNamedAttrs} from '../util';

export interface SoftmaxNodeConfig {
  axis?: onnx.AttributeProto;
}

export class Softmax extends OnnxNode {
  getTfjsLayerConfig(node: onnx.INodeProto, input?: SymbolicTensor[]):
      SoftmaxLayerConfig {
    const conf = getNamedAttrs(node.attribute) as SoftmaxNodeConfig;
    const axis = parseAttrOrDefault(conf.axis, 0) as number;
    const inShape = input[0].shape;

    return {
      axis: parseAxis(axis, inShape)
    }
  }

  getTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]): Layer {
    const conf = this.getTfjsConfig(node, input) as SoftmaxLayerConfig;
    return tf.layers.softmax(conf)
  }
}
