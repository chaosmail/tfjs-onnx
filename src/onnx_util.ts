import * as tf from '@tensorflow/tfjs';
import {Tensor} from '@tensorflow/tfjs';
import {DType, TypedArray} from '@tensorflow/tfjs-core/dist/types';
import {onnx} from 'onnx-proto';

export function parseAxis(axis: number, shape: number[]) {
  // convert to channelsLast
  switch (shape.length) {
    case 4:
      return axis == 1 ? 3 : axis;
    case 2:
      return axis == 1 ? 0 : axis;
    default:
      return axis;
  }
}

export function parseShape(shape: number[]) {
  // convert to channelsLast
  switch (shape.length) {
    case 4:
      return [shape[0], shape[3], shape[2], shape[1]];
    case 3:
      return [shape[2], shape[1], shape[0]];
    case 2:
      return [shape[1], shape[0]];
    default:
      return shape;
  }
}

export function parseAttrOrDefault(attr: onnx.AttributeProto, def?: any): any {
  return attr === undefined ? def : parseAttr(attr);
}

export function parseAttr(attr: onnx.AttributeProto): any {
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

export function parseTensorDtype(tensor: onnx.TensorProto): DType {
  switch (tensor.dataType) {
    case onnx.TensorProto.DataType.INT8:
      console.warn(`'Int8Array' type is not supported in tfjs. Converting to ${
          DType.int32}`);
      return DType.int32;
    case onnx.TensorProto.DataType.INT16:
      console.warn(`'Int16Array' type is not supported in tfjs. Converting to ${
          DType.int32}`);
      return DType.int32;
    case onnx.TensorProto.DataType.INT32:
      return DType.int32;
    case onnx.TensorProto.DataType.INT64:
      console.warn(
          `'Int64Array' type is not supported in JavaScript. Trying to convert to ${
              DType.int32}`);
      return DType.int32;
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

function getArrayBuffer(b: Uint8Array) {
  const data = new Uint8Array(b);
  return data.buffer;
}

export function parseTensorData(tensor: onnx.TensorProto): TypedArray {
  switch (tensor.dataType) {
    case onnx.TensorProto.DataType.INT8:
      return new Int32Array(new Int8Array(getArrayBuffer(tensor.rawData)));
    case onnx.TensorProto.DataType.INT16:
      return new Int32Array(new Int16Array(getArrayBuffer(tensor.rawData)));
    case onnx.TensorProto.DataType.INT32:
      return tensor.int32Data.length > 0 ?
          new Int32Array(tensor.int32Data) :
          new Int32Array(getArrayBuffer(tensor.rawData));
    case onnx.TensorProto.DataType.INT64:
      if (tensor.int64Data.length) {
        return new Int32Array(tensor.int32Data);
      }
      throw new Error(`'Int64Array' type not suppoert in JavaScript`);
    case onnx.TensorProto.DataType.FLOAT:
      return tensor.floatData.length > 0 ?
          new Float32Array(tensor.floatData) :
          new Float32Array(getArrayBuffer(tensor.rawData));
    case onnx.TensorProto.DataType.DOUBLE:
      return new Float32Array(new Float64Array(getArrayBuffer(tensor.rawData)));
    case onnx.TensorProto.DataType.UNDEFINED:
    default:
      throw new Error(`Cannot parse tensor '${tensor.dataType}'`);
  }
}

export function parseTensor(tensor: onnx.TensorProto): Tensor {
  const shape = tensor.dims as number[];
  const dtype = parseTensorDtype(tensor);
  const typedArray = parseTensorData(tensor);
  const data = tf.tensor(typedArray, shape, dtype);

  switch (shape.length) {
    case 4:
      return data.transpose([3, 2, 1, 0]);
    case 3:
      return data.transpose([2, 1, 0]);
    case 2:
      return data.transpose([1, 0]);
    default:
      return data;
  }
}
