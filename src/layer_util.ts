import * as tf from '@tensorflow/tfjs';
import {Shape, SymbolicTensor} from '@tensorflow/tfjs';
import {PaddingMode} from '@tensorflow/tfjs-layers/dist/common';
import {Layer, LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {onnx} from 'onnx-proto';

import {ConstantCompat} from './compat/core';
import {AutoPad, ConvNodeConfig} from './layers/convolution';
import {parseAttrOrDefault, parseShape} from './onnx_util';
import {getLayerName, getNamedAttrs} from './util';

export function getInputName(graph: onnx.IGraphProto) {
  return getLayerName(graph.input[graph.input.length - 1]);
}

export function input(name: string, shape: Shape): SymbolicTensor {
  const conf = {name: name, shape: shape};
  return tf.input(conf);
}

export function getInputShape(shape: number[]) {
  const outShape = parseShape(shape);
  // we need to remove the batch dimensions
  return outShape.length == 4 ? outShape.slice(1) : outShape;
}

export function isConstantLayer(layer: Layer) {
  return layer instanceof ConstantCompat && layer.outboundNodes.length > 0;
}

export function getCommonConfig(node: onnx.INodeProto): LayerConfig {
  return {name: node.name};
}

// TODO this code could be removed once we add compat/conv
export function getTfjsPadding(pads: number[], auto_pad: AutoPad): PaddingMode {
  const checkAutoPad = auto_pad !== null && auto_pad != 'VALID';
  const checkPads = pads !== null && pads.length > 0 && pads[0] != 0;
  return checkAutoPad || checkPads ? 'same' : 'valid'
}

// TODO this code could move to layers/conv
export function getConvDim(node: onnx.INodeProto): number {
  const conf = getNamedAttrs(node.attribute) as ConvNodeConfig;
  return parseAttrOrDefault(conf.kernel_shape, []).length || 2;
}
