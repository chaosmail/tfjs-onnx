import * as tf from '@tensorflow/tfjs';
import {SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {DenseLayerConfig, DropoutLayerConfig, ReshapeLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/core';
import {onnx} from 'onnx-proto';

import {ConstantCompat, ConstantLayerConfig, MatMulCompat} from '../compat/core';
import {OnnxNode, WeightInitializer} from '../node';
import {parseAttr, parseAttrOrDefault, parseShape, parseTensor} from '../onnx_util';
import {getNamedAttrs} from '../util';

export interface ConstantNodeConfig {
  value?: onnx.AttributeProto;
}

export class Constant extends OnnxNode {
  static getConstantAttr(node: onnx.INodeProto) {
    const conf = getNamedAttrs(node.attribute) as ConstantNodeConfig;
    const value = parseAttr(conf.value) as onnx.TensorProto;
    return parseTensor(value) as Tensor;
  }

  getTfjsLayerConfig(node: onnx.INodeProto): ConstantLayerConfig {
    const value = Constant.getConstantAttr(node);

    return {
      value: value.expandDims(0), inputShape: value.shape
    }
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node) as ConstantLayerConfig;
    return new ConstantCompat(conf);
  }
}

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

export class Flatten extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return tf.layers.flatten(conf);
  }
}

export interface ReshapeNodeConfig {
  shape?: onnx.AttributeProto;
}

export class Reshape extends OnnxNode {
  isSimplifiable(input?: SymbolicTensor[]) {
    if (input.length == 1 && input[0] !== undefined &&
        input[0].sourceLayer instanceof ConstantCompat) {
      return true;
    }
    return false;
  }

  getTfjsLayerConfig(node: onnx.INodeProto, input?: SymbolicTensor[]):
      ReshapeLayerConfig {
    const conf = getNamedAttrs(node.attribute) as ReshapeNodeConfig;
    const value = parseAttr(conf.shape);
    const shape = parseShape(value);
    // Add batch dimension if required
    const targetShape =
        input[0].shape[0] == null ? shape : [null].concat(shape);
    return {targetShape: targetShape};
  }

  getTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]): Layer {
    const conf = this.getTfjsConfig(node, input) as ReshapeLayerConfig;

    // TODO not only reshape can take constant inputs
    // check if this can be generalized to all layers
    if (this.isSimplifiable(input)) {
      const contsNode = this.model.nodes[node.input[0]];
      const constValue = Constant.getConstantAttr(contsNode);
      const shape = conf.targetShape;
      const value = constValue.reshape(shape).expandDims(0);
      const constConf = {name: conf.name, value: value, inputShape: shape};
      return new ConstantCompat(constConf);
    }

    return tf.layers.reshape(conf);
  }
}

export class MatMul extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    const conf = this.getTfjsConfig(node);
    return new MatMulCompat(conf);
  }
}
