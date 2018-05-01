import * as tf from '@tensorflow/tfjs';
import {Tensor} from '@tensorflow/tfjs';
import {DType, TypedArray} from '@tensorflow/tfjs-core/dist/types';
import {LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {onnx} from 'onnx-proto';

export function normalizeArrayToObj<T>(
    array: T[], indexKey: keyof T): {[key: string]: T} {
  const normalizedObject: any = {};
  for (let i = 0; i < array.length; i++) {
    const key = array[i][indexKey];
    normalizedObject[key] = array[i];
  }
  return normalizedObject as {
    [key: string]: T
  }
}

export function joinArraysToObj<T>(
    keys: string[], values: T[]): {[key: string]: T} {
  const normalizedObject: any = {};
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    normalizedObject[key] = values[i];
  }
  return normalizedObject as {
    [key: string]: T
  }
}

// tslint:disable-next-line:no-any
export function isNotNull(val: any): boolean {
  return val !== undefined && val !== null;
}

export function fetchText(uri: string): Promise<string> {
  return fetch(new Request(uri))
      .then(handleFetchErrors)
      .then((res) => res.text());
}

export function fetchArrayBuffer(uri: string): Promise<ArrayBuffer> {
  return fetch(new Request(uri))
      .then(handleFetchErrors)
      .then((res) => res.arrayBuffer());
}

function handleFetchErrors(response: Response) {
  if (!response.ok) {
    throw Error(response.statusText);
  }
  return response;
}

export function parseOnnxAxis(axis: number, shape: number[]) {
  // convert to channelsLast
  return shape.length == 4 && axis == 1 ? 3 : axis;
}

export function parseOnnxShape(shape: number[]) {
  const [b, c, w, h] = shape;
  // convert to channelsLast
  return shape.length == 4 ? [b, w, h, c] : shape;
}

export function parseOnnxModel(data: ArrayBuffer) {
  return onnx.ModelProto.decode(new Uint8Array(data));
}

export function parseAttrOrDefault(attr: onnx.AttributeProto, def?: any): any {
  return attr === undefined ? def : parseOnnxAttr(attr);
}

export function parseOnnxAttr(attr: onnx.AttributeProto): any {
  switch (attr.type) {
    case onnx.AttributeProto.AttributeType.FLOAT:
      return attr.f;
    case onnx.AttributeProto.AttributeType.INT:
      return attr.i;
    case onnx.AttributeProto.AttributeType.STRING:
      return attr.s;
    case onnx.AttributeProto.AttributeType.TENSOR:
      return attr.t;
    case onnx.AttributeProto.AttributeType.GRAPH:
      return attr.g;
    case onnx.AttributeProto.AttributeType.FLOATS:
      return attr.floats;
    case onnx.AttributeProto.AttributeType.INTS:
      return attr.ints;
    case onnx.AttributeProto.AttributeType.STRINGS:
      return attr.strings;
    case onnx.AttributeProto.AttributeType.TENSORS:
      return attr.tensors;
    case onnx.AttributeProto.AttributeType.GRAPHS:
      return attr.graphs;
    case onnx.AttributeProto.AttributeType.UNDEFINED:
    default:
      throw new Error(`Cannot parse attr '${attr.name}'`);
  }
}

export function onnxTensorTypeToTfjsDtype(tensor: onnx.TensorProto): DType {
  switch (tensor.dataType) {
    case onnx.TensorProto.DataType.INT8:
      throw new Error(`Cannot use 'int8' tensor in tfjs`);
    case onnx.TensorProto.DataType.INT16:
      throw new Error(`Cannot use 'int16' tensor in tfjs`);
    case onnx.TensorProto.DataType.INT32:
      return DType.int32;
    case onnx.TensorProto.DataType.INT64:
      throw new Error(`'Int64Array' type not suppoert in JavaScript`);
    case onnx.TensorProto.DataType.UINT8:
      throw new Error(`Cannot use 'uint8' tensor in tfjs`);
    case onnx.TensorProto.DataType.UINT16:
      throw new Error(`Cannot use 'uint16' tensor in tfjs`);
    case onnx.TensorProto.DataType.UINT32:
      throw new Error(`Cannot use 'uint32' tensor in tfjs`);
    case onnx.TensorProto.DataType.UINT16:
      throw new Error(`Cannot use 'uint64' tensor in tfjs`);
    case onnx.TensorProto.DataType.FLOAT:
      return DType.float32;
    case onnx.TensorProto.DataType.DOUBLE:
      console.warn(`'double' type is not supported in tfjs. Converting to ${
          DType.float32}`)
      return DType.float32;
    case onnx.TensorProto.DataType.UNDEFINED:
    default:
      throw new Error(`Cannot parse tensor '${tensor.dataType}'`);
  }
}

export function parseOnnxTensor(tensor: onnx.TensorProto): TypedArray {
  const getArrayBuffer =
      (b: Uint8Array) => {
        const data = new Uint8Array(b);
        return data.buffer;
      }

  switch (tensor.dataType) {
    case onnx.TensorProto.DataType.INT32:
      return new Int32Array(getArrayBuffer(tensor.rawData));
    case onnx.TensorProto.DataType.FLOAT:
      return new Float32Array(getArrayBuffer(tensor.rawData));
    case onnx.TensorProto.DataType.DOUBLE:
      return new Float32Array(new Float64Array(getArrayBuffer(tensor.rawData)));
    case onnx.TensorProto.DataType.UNDEFINED:
    default:
      throw new Error(`Cannot parse tensor '${tensor.dataType}'`);
  }
}

export function onnxShapeToTfjsShape(shape: onnx.ITensorShapeProto): number[] {
  return shape.dim.map(d => d.dimValue) as number[];
}

export function getNamedAttrs<T>(attrs?: any[]): T {
  return <any>normalizeArrayToObj(attrs, 'name') as T;
}

export function getTfjsCommonConfig(node: onnx.INodeProto): LayerConfig {
  return {name: node.name};
}

export async function loadOnnxModel(modelUrl: string):
    Promise<onnx.IModelProto> {
  const buffer = await fetchArrayBuffer(modelUrl);
  return await parseOnnxModel(buffer);
}

export function getBlobValues(graph: onnx.IGraphProto) {
  const blobs = graph.initializer;
  const weights = blobs.map(onnxTensorToTfjsWeigths);
  const names = blobs.map(d => d.name);
  return joinArraysToObj(names, weights);
}

export function getValueInfo(valueInfo: onnx.IValueInfoProto[]) {
  const shapes =
      valueInfo.map(d => d.type.tensorType.shape).map(onnxShapeToTfjsShape);
  const names = valueInfo.map(d => d.name);
  return joinArraysToObj(names, shapes);
}

export function getBlobShapes(graph: onnx.IGraphProto) {
  // Parse input/output tensor shapes
  const inputShapes = getValueInfo(graph.input);
  const outputShapes = getValueInfo(graph.output);
  return Object.assign({}, inputShapes, outputShapes);
}

export function getInputShape(shape: number[]) {
  // we need to remove the batch dimensions
  const [channels, width, height] = shape.length == 4 ? shape.slice(1) : shape;
  return [height, width, channels];
}

export function getLayerName(node: onnx.INodeProto) {
  return node.name ? node.name : node.output[0];
}

export function onnxTensorToTfjsWeigths(tensor: onnx.TensorProto): Tensor {
  const shape = tensor.dims as number[];
  const dtype = onnxTensorTypeToTfjsDtype(tensor);
  const typedArray = parseOnnxTensor(tensor);
  const data = tf.tensor(typedArray, shape, dtype);
  return shape.length == 4 ? data.transpose([3, 2, 1, 0]) : data;
}

export function loadImageData(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  return new Promise((resolve, reject) => {
    img.crossOrigin = 'anonymous';
    img.src = url;
    img.onload = () => resolve(img);
    img.onerror = reject;
  });
}
