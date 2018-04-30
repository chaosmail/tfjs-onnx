import * as tf from '@tensorflow/tfjs';
import {PaddingMode} from '@tensorflow/tfjs-layers/dist/common';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ConvLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/convolutional';
import {onnx} from 'onnx-proto';

import {OnnxNode, WeightInitializer} from '../node';
import {getNamedAttrs, parseAttrOrDefault} from '../util';

export type AutoPad = 'SAME_UPPER'|'SAME_LOWER'|'VALID';

export interface ConvNodeConfig {
  auto_pad?: onnx.AttributeProto;
  dilations?: onnx.AttributeProto;
  group?: onnx.AttributeProto;
  kernel_shape?: onnx.AttributeProto;
  pads?: onnx.AttributeProto;
  strides?: onnx.AttributeProto;
}

export class Conv extends OnnxNode {
  static getTfjsPadding(pads: number[], auto_pad: AutoPad): PaddingMode {
    const checkAutoPad = auto_pad !== null && auto_pad != 'VALID';
    const checkPads = pads !== null && pads.length > 0 && pads[0] != 0;
    return checkAutoPad || checkPads ? 'same' : 'valid'
  }

  static getConvDim(node: onnx.INodeProto): number {
    const conf = getNamedAttrs(node.attribute) as ConvNodeConfig;
    return parseAttrOrDefault(conf.kernel_shape, []).length || 2;
  }

  getTfjsLayerConfig(node: onnx.INodeProto): ConvLayerConfig {
    const conf = getNamedAttrs(node.attribute) as ConvNodeConfig;
    const kernelSize = parseAttrOrDefault(conf.kernel_shape) as number[];
    const strides = parseAttrOrDefault(conf.strides, 1) as number[];
    const pads = parseAttrOrDefault(conf.pads, null);
    const autoPad = parseAttrOrDefault(conf.auto_pad, null);
    const padding = Conv.getTfjsPadding(pads, autoPad);
    const dilationRate = parseAttrOrDefault(conf.dilations, 1);

    const w = node.input[1];
    const b = node.input[2];
    const weightShape = this.model.blobShapes[w];
    const filters = weightShape[0];
    const kernel = this.model.blobValues[w];
    const bias = this.model.blobValues[b];

    return {
      kernelSize: kernelSize, strides: strides, padding: padding,
          dilationRate: dilationRate, filters: filters,
          kernelInitializer: new WeightInitializer(kernel),
          biasInitializer: new WeightInitializer(bias),
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Conv.getConvDim(node);
    const conf = this.getTfjsConfig(node) as ConvLayerConfig;
    return dim == 1 ? tf.layers.conv1d(conf) : tf.layers.conv2d(conf);
  }
}
