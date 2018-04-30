import * as tf from '@tensorflow/tfjs';
import {ActivationIdentifier} from '@tensorflow/tfjs-layers/dist/activations';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {ActivationLayerConfig} from '@tensorflow/tfjs-layers/dist/layers/core';
import {onnx} from 'onnx-proto';

import {OnnxNode} from '../node';

export abstract class Activation extends OnnxNode {
  getTfjsLayer(node: onnx.INodeProto): Layer {
    return tf.layers.activation(
        this.getTfjsConfig(node) as ActivationLayerConfig)
  }
}

export class Relu extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'relu' as ActivationIdentifier};
  }
}

export class Tanh extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'tanh' as ActivationIdentifier};
  }
}

export class Sigmoid extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'sigmoid' as ActivationIdentifier};
  }
}

export class Elu extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'elu' as ActivationIdentifier};
  }
}

export class Softplus extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'softplus' as ActivationIdentifier};
  }
}

export class Softsign extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'softsign' as ActivationIdentifier};
  }
}

export class HardSigmoid extends Activation {
  getTfjsLayerConfig(node: onnx.INodeProto): ActivationLayerConfig {
    return {activation: 'hardsigmoid' as ActivationIdentifier};
  }
}
