import * as tf from '@tensorflow/tfjs';
import {Model, SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {ContainerConfig, Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {onnx} from 'onnx-proto';

import {Elu, HardSigmoid, Relu, Sigmoid, Softplus, Softsign, Tanh} from './layers/activations'
import {Softmax} from './layers/advanced_activations';
import {Conv} from './layers/convolution'
import {Dense, Dropout, Flatten, Reshape} from './layers/core';
import {Add, Concat, Mul} from './layers/merge';
import {AveragePool, GlobalAveragePool, GlobalMaxPool, MaxPool} from './layers/pooling';
import {OnnxNode} from './node';
import * as util from './util';

type NodeFactory = {
  [key: string]: typeof OnnxNode
};

const nodeFactory: NodeFactory = {
  'Add': Add,
  'AveragePool': AveragePool,
  'Concat': Concat,
  'Conv': Conv,
  'Dropout': Dropout,
  'Elu': Elu,
  'FC': Dense,
  'Flatten': Flatten,
  'GlobalAveragePool': GlobalAveragePool,
  'GlobalMaxPool': GlobalMaxPool,
  'HardSigmoid': HardSigmoid,
  'MaxPool': MaxPool,
  'Mul': Mul,
  'Relu': Relu,
  'Reshape': Reshape,
  'Sigmoid': Sigmoid,
  'Softmax': Softmax,
  'Softplus': Softplus,
  'Softsign': Softsign,
  'Tanh': Tanh,
};

export async function load(modelUrl: string): Promise<Model> {
  const model = new OnnxModel(modelUrl);
  await model.load();
  return model.getModel();
}

export class OnnxModel {
  onnx: onnx.IModelProto;
  graph: onnx.IGraphProto;
  blobShapes: {[name: string]: number[]};
  blobValues: {[name: string]: Tensor};
  layers: {[name: string]: Layer};
  blobs: {[name: string]: Tensor|SymbolicTensor};

  constructor(public modelUrl: string) {}

  async load() {
    this.onnx = await util.loadOnnxModel(this.modelUrl);
    this.graph = this.onnx.graph;
    this.blobShapes = util.getBlobShapes(this.graph);
    this.blobValues = util.getBlobValues(this.graph);
    this.layers = this.getLayers(this.graph);
  }

  getModel() {
    const outputNames = this.graph.output.map(d => d.name);
    const inputName = this.graph.input[this.graph.input.length - 1].name;
    const inputConf = {shape: util.getInputShape(this.blobShapes[inputName])};
    const input = tf.input(inputConf);

    this.blobs = {};
    this.blobs[inputName] = input;

    for (let i = 0; i < this.graph.node.length; ++i) {
      let currNode = this.graph.node[i];
      let inputBlobs: SymbolicTensor[] = [];
      let currLayerName = currNode.output[0];
      let currLayer = this.layers[currLayerName];

      for (let j = 0; j < currNode.input.length; ++j) {
        let inputNodeName = currNode.input[j];
        if (this.blobs.hasOwnProperty(inputNodeName)) {
          inputBlobs.push(this.blobs[inputNodeName] as SymbolicTensor);
        }
      }

      if (inputBlobs.length > 0) {
        this.blobs[currLayerName] =
            currLayer.apply(inputBlobs) as SymbolicTensor;
      }
    }

    // Finally we can select the output blobs
    const outputs = outputNames.map(d => this.blobs[d]);

    // Create the container config
    const config = {inputs: input, outputs: outputs, name: this.graph.name} as
        ContainerConfig;

    // Create the model
    return tf.model(config);
  }

  getLayers(graph: onnx.IGraphProto) {
    const nodes = graph.node;
    const layers = nodes.map(d => this.getTfjsLayer(d));
    const names = nodes.map(d => d.output[0]);
    return util.joinArraysToObj(names, layers);
  }

  getTfjsLayer(node: onnx.INodeProto): Layer {
    if (nodeFactory.hasOwnProperty(node.opType)) {
      const onnxNode = (<any>nodeFactory[node.opType]).from(this);
      return onnxNode.getTfjsLayer(node);
    }
    throw new Error(`'${node.opType}' is not implemented in tfjs-onnx.`);
  }
}
