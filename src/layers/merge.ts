import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ConcatenateLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/merge';
import {onnx} from 'onnx-proto';

import {DivCompat, SubCompat} from '../compat/merge';
import {OnnxNode} from '../node';
import {parseAttrOrDefault, parseAxis} from '../onnx_util';
import {getNamedAttrs} from '../util';

export interface ConcatNodeConfig {
  axis?: onnx.AttributeProto;
}

export class Concat extends OnnxNode {
  getTfjsLayerConfig(node: onnx.INodeProto, input?: SymbolicTensor[]):
      ConcatenateLayerConfig {
    const conf = getNamedAttrs(node.attribute) as ConcatNodeConfig;
    const axis = parseAttrOrDefault(conf.axis, 0) as number;
    const inShape = input[0].shape;

    return {
      axis: parseAxis(axis, inShape)
    }
  }

  getTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]): Layer {
    const conf = this.getTfjsConfig(node, input) as ConcatenateLayerConfig;
    return tf.layers.concatenate(conf)
  }
}

export class Add extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return tf.layers.add(conf);
  }
}

export class Sub extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return new SubCompat(conf);
  }
}

export class Mul extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return tf.layers.multiply(conf);
  }
}

export class Div extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return new DivCompat(conf);
  }
}
