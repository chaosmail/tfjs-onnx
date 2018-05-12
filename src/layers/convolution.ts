import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ConvLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/convolutional';
import {onnx} from 'onnx-proto';

import {getConvDim, getTfjsPadding} from '../layer_util';
import {OnnxNode, WeightInitializer} from '../node';
import {parseAttrOrDefault} from '../onnx_util';
import {getNamedAttrs} from '../util';

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
    const padding = getTfjsPadding(pads, autoPad);
    const dilationRate = parseAttrOrDefault(conf.dilations, 1);

    // tfjs shape: numChannels, inHeight, inWidth, inChannels
    // conv shape: inHeight, inWidth, inChannels, numChannels
    const kernel = this.getTensorAttr(node.input[1]).transpose([2, 1, 3, 0]);
    const bias = node.input[2] ? this.getTensorAttr(node.input[2]) : null;
    const filters = kernel.shape[3];

    return {
      kernelSize: kernelSize, strides: strides, padding: padding,
          dilationRate: dilationRate, filters: filters,
          kernelInitializer: new WeightInitializer(kernel),
          useBias: Boolean(bias),
          biasInitializer: bias ? new WeightInitializer(bias) : undefined,
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const dim = getConvDim(node);
    const conf = this.getTfjsConfig(node) as ConvLayerConfig;
    return dim == 1 ? tf.layers.conv1d(conf) : tf.layers.conv2d(conf);
  }

  prepareInput(input?: SymbolicTensor[]) {
    return input.slice(0, 1);
  }
}
