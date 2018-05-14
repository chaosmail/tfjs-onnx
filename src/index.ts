import {loadModel} from './model';
import * as util from './util';

export * from '@tensorflow/tfjs';

export const onnx = {
  'util': util,
  'loadModel': loadModel
};
