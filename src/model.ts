import {Model, ModelPredictConfig, SymbolicTensor, Tensor} from '@tensorflow/tfjs';
import {ContainerConfig, Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {onnx} from 'onnx-proto';

import {ConstantCompat} from './compat/core';
import * as layer_util from './layer_util';
import {Elu, HardSigmoid, Relu, Sigmoid, Softplus, Softsign, Tanh} from './layers/activations'
import {Softmax} from './layers/advanced_activations';
import {Conv} from './layers/convolution'
import {Constant, Dense, Dropout, Flatten, MatMul, Reshape} from './layers/core';
import {Add, Concat, Div, Mul, Sub} from './layers/merge';
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
  'Constant': Constant,
  'Conv': Conv,
  'Div': Div,
  'Dropout': Dropout,
  'Elu': Elu,
  'FC': Dense,
  'Flatten': Flatten,
  'GlobalAveragePool': GlobalAveragePool,
  'GlobalMaxPool': GlobalMaxPool,
  'HardSigmoid': HardSigmoid,
  'MaxPool': MaxPool,
  'MatMul': MatMul,
  'Mul': Mul,
  'Relu': Relu,
  'Reshape': Reshape,
  'Sigmoid': Sigmoid,
  'Softmax': Softmax,
  'Softplus': Softplus,
  'Softsign': Softsign,
  'Sub': Sub,
  'Tanh': Tanh,
};

export type Blob = Tensor|SymbolicTensor;

export async function loadModel(modelUrl: string): Promise<ModelCompat> {
  const model = new OnnxModel(modelUrl);
  await model.load();
  return model.getModel();
}

export async function loadModelFromBuffer(modelBuffer: ArrayBuffer): Promise<ModelCompat> {
  const model = new OnnxModel('');
  await model.loadFromBuffer(modelBuffer);
  return model.getModel();
}

export class ModelCompat extends Model {
  constructor(config: ContainerConfig, public onnx: OnnxModel) {
    super(config);
  }

  predict(x: Tensor|Tensor[], config: ModelPredictConfig = {}): Tensor
      |Tensor[] {
    return super.predict(this.onnx.getAllInputs(x), config);
  }
}

export class OnnxModel {
  onnx: onnx.IModelProto;
  graph: onnx.IGraphProto;
  blobShapes: {[name: string]: number[]};
  blobValues: {[name: string]: Tensor};
  nodes: {[name: string]: onnx.INodeProto};
  layers: {[name: string]: Layer} = {};
  blobs: {[name: string]: Blob} = {};

  constructor(public modelUrl: string) {}

  async load() {
    this.onnx = await util.loadOnnxModel(this.modelUrl);
    this.graph = this.onnx.graph;
    this.nodes = util.getNodes(this.graph);
    this.blobShapes = util.getBlobShapes(this.graph);
    this.blobValues = util.getBlobValues(this.graph);
  }

  async loadFromBuffer(model: ArrayBuffer) {
    this.onnx = await util.parseOnnxModel(model);
    this.graph = this.onnx.graph;
    this.nodes = util.getNodes(this.graph);
    this.blobShapes = util.getBlobShapes(this.graph);
    this.blobValues = util.getBlobValues(this.graph);
  }

  getAllInputs(input: Tensor | Tensor[]): Tensor[] {
    return [].concat(input, this.getConstantInputs());
  }

  getModel() {
    const input = this.getInputLayer();

    this.blobs[input.name] = input;

    for (let i = 0; i < this.graph.node.length; ++i) {
      let currNode = this.graph.node[i];

      let inputBlobs: SymbolicTensor[] = [];
      for (let j = 0; j < currNode.input.length; ++j) {
        let inputNodeName = currNode.input[j];
        if (this.blobs.hasOwnProperty(inputNodeName)) {
          inputBlobs.push(this.blobs[inputNodeName] as SymbolicTensor);
        }
      }

      const [layer, output] = this.setupTfjsLayer(currNode, inputBlobs);
      this.layers[layer.name] = layer;

      currNode.output.forEach((d) => {
        this.blobs[d] = output;
      });
    }

    // Select the input blobs
    const inputs = [input].concat(this.getSymbolicConstantInputs());

    // Select the output blobs
    const outputs = this.getSymbolicOutputs();

    // Create the container config
    const config = {inputs: inputs, outputs: outputs, name: this.graph.name} as
        ContainerConfig;

    // Create the model
    return new ModelCompat(config, this);
  }

  setupTfjsLayer(node: onnx.INodeProto, input?: SymbolicTensor[]):
      [Layer, SymbolicTensor] {
    if (nodeFactory.hasOwnProperty(node.opType)) {
      const onnxNode = (<any>nodeFactory[node.opType]).from(this);
      return onnxNode.setup(node, input) as [Layer, SymbolicTensor];
    }
    throw new Error(`'${node.opType}' is not implemented in tfjs-onnx.`);
  }

  private getInputLayer(): SymbolicTensor {
    const name = layer_util.getInputName(this.graph);
    const shape = layer_util.getInputShape(this.blobShapes[name]);
    return layer_util.input(name, shape);
  }

  private getConstantInputNames(): string[] {
    return Object.keys(this.layers)
        .filter(d => layer_util.isConstantLayer(this.layers[d]));
  }

  private getConstantInputs(): Tensor[] {
    return this.getConstantInputNames()
               .map(d => this.layers[d] as ConstantCompat)
               .map(d => d.value) as Tensor[];
  }

  private getSymbolicConstantInputs() {
    return this.getConstantInputNames()
               .map(d => this.nodes[d].output[0])
               .map(d => this.blobs[d]) as SymbolicTensor[];
  }

  private getSymbolicOutputs() {
    return this.graph.output.map(d => d.name).map(d => this.blobs[d]);
  }
}
