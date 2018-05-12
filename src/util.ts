import {onnx} from 'onnx-proto';

import {Tensor} from '.';
import {parseTensor} from './onnx_util';

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

export function parseOnnxModel(data: ArrayBuffer) {
  return onnx.ModelProto.decode(new Uint8Array(data));
}

export function getNamedAttrs<T>(attrs?: any[]): T {
  return <any>normalizeArrayToObj(attrs, 'name') as T;
}

export async function loadOnnxModel(modelUrl: string):
    Promise<onnx.IModelProto> {
  const buffer = await fetchArrayBuffer(modelUrl);
  return await parseOnnxModel(buffer);
}

export function getBlobValues(graph: onnx.IGraphProto):
    {[name: string]: Tensor} {
  const blobs = graph.initializer;
  const weights = blobs.map(parseTensor);
  const names = blobs.map(getLayerName);
  return joinArraysToObj(names, weights);
}

export function getValueInfo(valueInfo: onnx.IValueInfoProto[]):
    {[name: string]: number[]} {
  const getDimValues = (shape: onnx.ITensorShapeProto) =>
      shape.dim.map(d => d.dimValue) as number[];

  const shapes = valueInfo.map(d => d.type.tensorType.shape).map(getDimValues);
  const names = valueInfo.map(getLayerName);
  return joinArraysToObj(names, shapes);
}

export function getBlobShapes(graph: onnx.IGraphProto) {
  // Parse input/output tensor shapes
  const inputShapes = getValueInfo(graph.input);
  const outputShapes = getValueInfo(graph.output);
  return Object.assign({}, inputShapes, outputShapes);
}

export function getLayerName(node: onnx.INodeProto) {
  return node.name ? node.name : node.output[0];
}

export function getNodes(graph: onnx.IGraphProto) {
  const nodes = graph.node;
  const names = nodes.map(getLayerName);
  return joinArraysToObj(names, nodes);
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
