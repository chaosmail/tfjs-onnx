import * as tf from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ConcatenateLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/merge';
import {onnx} from 'onnx-proto';

import {OnnxNode} from '../node';
import {getNamedAttrs, parseAttrOrDefault} from '../util';

export interface ConcatNodeConfig {
  axis?: onnx.AttributeProto;
}

export class Concat extends OnnxNode {
  static parseAxis(axis: number) {
    // TODO apply only for shape.length === 4
    return axis == 1 ? 3 : axis;
  }

  getTfjsLayerConfig(node: onnx.INodeProto): ConcatenateLayerConfig {
    const conf = getNamedAttrs(node.attribute) as ConcatNodeConfig;
    const axis = parseAttrOrDefault(conf.axis, 0) as number;
    return {
      axis: Concat.parseAxis(axis)
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node) as ConcatenateLayerConfig;
    return tf.layers.concatenate(conf)
  }
}
