import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {PaddingMode} from '@tensorflow/tfjs-layers/dist/common';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ConvLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/convolutional';
import {onnx} from 'onnx-proto';

import {OnnxNode, WeightInitializer} from '../node';
import {getNamedAttrs, parseAttrOrDefault} from '../util';

import {Constant} from './core';

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

  getTensorAttr(name: string): Tensor {
    if (this.model.blobValues !== undefined &&
        this.model.blobValues.hasOwnProperty(name)) {
      return this.model.blobValues[name];
    } else if (
        this.model.nodes !== undefined &&
        this.model.nodes.hasOwnProperty(name)) {
      const node = this.model.nodes[name];
      if (node.opType == 'Constant') {
        return Constant.getConstantAttr(node);
      }
      throw new Error(`Cannot extract tensor attribute '${
          name}' from layer other than 'Constant'`);
    }
    // TODO if model is not trained, we can use the
    // this.model.blobShapes to extract tensor shape
    else {
      throw new Error(`Cannot find tensor attribute '${name}'`);
    }
  }

  getTfjsLayerConfig(node: onnx.INodeProto): ConvLayerConfig {
    const conf = getNamedAttrs(node.attribute) as ConvNodeConfig;
    const kernelSize = parseAttrOrDefault(conf.kernel_shape) as number[];
    const strides = parseAttrOrDefault(conf.strides, 1) as number[];
    const pads = parseAttrOrDefault(conf.pads, null);
    const autoPad = parseAttrOrDefault(conf.auto_pad, null);
    const padding = Conv.getTfjsPadding(pads, autoPad);
    const dilationRate = parseAttrOrDefault(conf.dilations, 1);

    const kernel = this.getTensorAttr(node.input[1]);
    const bias = node.input[2] ? this.getTensorAttr(node.input[2]) : null;
    const filters = kernel.shape[kernel.shape.length - 1];

    return {
      kernelSize: kernelSize, strides: strides, padding: padding,
          dilationRate: dilationRate, filters: filters,
          kernelInitializer: new WeightInitializer(kernel),
          useBias: Boolean(bias),
          biasInitializer: bias ? new WeightInitializer(bias) : undefined,
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = Conv.getConvDim(node);
    const conf = this.getTfjsConfig(node) as ConvLayerConfig;
    return dim == 1 ? tf.layers.conv1d(conf) : tf.layers.conv2d(conf);
  }

  prepareInput(input?: SymbolicTensor[]) {
    return input.slice(0, 1);
  }
}
